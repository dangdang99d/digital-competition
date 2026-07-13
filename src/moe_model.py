"""E33 model-level MoE over a ModernBERT (granite) backbone.

Architecture (experiments/moe/results.md, pinned 2026-07-13):
  shared embeddings -> shared trunk (granite layers 0..k-1) -> {gate, 4 expert stacks}
  - gate: mask-aware mean-pool of the (un-final-normed) trunk hidden states,
    LayerNorm, concat structure scalars (log1p hist-count, first-step flag),
    MLP -> softmax over experts. Pre-expert routing (Jacobs g(x); Switch/Mixtral
    route on the hidden state entering the experts).
  - expert e: granite layers k..21 (own copy) -> final_norm -> CLS/mean pool per
    config.classifier_pooling -> ModernBertPredictionHead -> Linear(hidden, C).
  - blend: gate-weighted mean of expert softmax probs (the trio's mechanism with
    learned per-row weights).

Loss = LS-CE on the blended distribution
     + balance_coef * load-balance aux (Switch eq. 4-6; diffed against HF
       transformers 4.51.3 `mixtral.load_balancing_loss_func`: identical with
       top_k=1 and one routing decision per SEQUENCE instead of per token —
       f_e = mean one-hot(argmax gate), P_e = mean gate prob, aux = E * sum f*P)
     + entropy_coef * batch-usage entropy floor: relu(entropy_tau*ln(E) - H(mean
       gate)) — blocks global collapse onto one expert (E29-4 failure mode)
       while leaving per-row decisiveness free (specialization is the point).

Symmetry breaking: granite's config has ALL dropouts = 0.0, so 4 byte-identical
experts under a fresh gate receive near-identical gradients and diverge only
through the gate's tiny init asymmetry. Each expert's classifier Linear is
therefore re-initialized with a distinct derived seed (init_seed*1000 + e).
Encoder stacks + head dense/norm stay byte-identical pretrained copies. (E12's
shared-init rule is about weight-space averaging of soup members; nothing here
is ever weight-averaged, and un-broken symmetry is the degenerate case.)

Only sdpa/eager attention is supported (the flash path unpads/repads inside
ModernBertModel.forward; the trunk/expert split would have to thread that
state through — not worth it, champion runs train on sdpa anyway).
"""
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MOE_CONFIG_NAME = "moe_config.json"
MOE_WEIGHTS_NAME = "moe_state.pt"
N_GATE_SCALARS = 2  # [log1p(n_history_events), is_first_step]


class GateNet(nn.Module):
    """Controller: pooled trunk states + structure scalars -> expert logits."""

    def __init__(self, hidden, gate_hidden, n_experts, n_scalars=N_GATE_SCALARS):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + n_scalars, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, n_experts),
        )
        self.n_scalars = n_scalars

    def forward(self, pooled, scalars):
        x = self.norm(pooled)
        if scalars is None:  # tolerate missing feats (e.g. quick probes)
            scalars = x.new_zeros(x.shape[0], self.n_scalars)
        x = torch.cat([x, scalars.to(dtype=x.dtype)], dim=-1)
        return self.mlp(x)  # (B, E) logits


class ExpertBranch(nn.Module):
    """Layers k..L-1 + final_norm + prediction head + classifier for one expert."""

    def __init__(self, layers, final_norm, head, classifier, drop_p):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.head = head
        self.drop = nn.Dropout(drop_p)
        self.classifier = classifier


class GraniteMoE(nn.Module):
    """Shared-trunk soft-gated MoE built by splitting one pretrained
    ModernBertForSequenceClassification. Consumes the loaded model in place
    (its .model.layers is truncated to the trunk)."""

    def __init__(self, base_scls, trunk_k=6, n_experts=4, gate_hidden=128,
                 init_seed=42, label_smoothing=0.1, balance_coef=0.01,
                 entropy_coef=0.01, entropy_tau=0.75):
        super().__init__()
        cfg = base_scls.config
        assert cfg.model_type == "modernbert", \
            f"GraniteMoE is implemented for ModernBERT only (got {cfg.model_type})"
        assert cfg._attn_implementation != "flash_attention_2", \
            "GraniteMoE supports sdpa/eager only (flash unpads inside forward)"
        depth = len(base_scls.model.layers)
        assert 0 < trunk_k < depth, f"trunk_k={trunk_k} out of range for depth {depth}"

        self.config = cfg                      # Trainer/inspection convenience
        self.trunk_k = trunk_k
        self.n_experts = n_experts
        self.num_labels = cfg.num_labels
        self.label_smoothing = label_smoothing
        self.balance_coef = balance_coef
        self.entropy_coef = entropy_coef
        self.entropy_tau = entropy_tau
        self.gradient_checkpointing = False

        full_layers = list(base_scls.model.layers)
        # expert branches FIRST (they deepcopy layers k.. before base is truncated)
        experts = []
        for e in range(n_experts):
            classifier = copy.deepcopy(base_scls.classifier)
            gen = torch.Generator().manual_seed(init_seed * 1000 + e)
            with torch.no_grad():
                classifier.weight.copy_(torch.empty_like(classifier.weight).normal_(
                    mean=0.0, std=0.02, generator=gen))
                if classifier.bias is not None:
                    classifier.bias.zero_()
            experts.append(ExpertBranch(
                layers=nn.ModuleList(copy.deepcopy(l) for l in full_layers[trunk_k:]),
                final_norm=copy.deepcopy(base_scls.model.final_norm),
                head=copy.deepcopy(base_scls.head),
                classifier=classifier,
                drop_p=cfg.classifier_dropout,
            ))
        self.experts = nn.ModuleList(experts)

        # trunk = the base ModernBertModel, truncated; final_norm moves into the
        # experts (each branch normalizes its OWN residual stream, as in the
        # dense model), so the trunk emits the raw layer-k hidden states.
        base_scls.model.layers = nn.ModuleList(full_layers[:trunk_k])
        base_scls.model.final_norm = nn.Identity()
        self.trunk = base_scls.model

        self.gate = GateNet(cfg.hidden_size, gate_hidden, n_experts)

    # ---- HF Trainer hooks (targs.gradient_checkpointing=True calls this) ----
    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    # ---- forward pieces ----
    def _run_layers(self, layers, h, attn4d, swm, pos):
        for layer in layers:
            if self.gradient_checkpointing and self.training:
                out = torch.utils.checkpoint.checkpoint(
                    layer.__call__, h, attn4d, swm, pos, use_reentrant=False)
            else:
                out = layer(h, attention_mask=attn4d,
                            sliding_window_mask=swm, position_ids=pos)
            h = out[0]
        return h

    def _encode(self, input_ids, attention_mask, gate_feats):
        """Embeddings + masks once, trunk once, gate once, all experts.
        Returns (gate_logprobs (B,E), expert_logits (B,E,C))."""
        B, L = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=device, dtype=torch.bool)
        pos = torch.arange(L, device=device).unsqueeze(0)
        # identical mask semantics to ModernBertModel.forward (sdpa/eager path)
        attn4d, swm = self.trunk._update_attention_mask(
            attention_mask, output_attentions=False)

        h = self.trunk.embeddings(input_ids=input_ids)
        h_trunk = self._run_layers(self.trunk.layers, h, attn4d, swm, pos)

        # controller: mask-aware mean-pool of the trunk stream + scalars
        m = attention_mask.to(h_trunk.dtype).unsqueeze(-1)
        pooled_gate = (h_trunk * m).sum(1) / m.sum(1).clamp(min=1.0)
        gate_logits = self.gate(pooled_gate, gate_feats).float()
        gate_logprobs = F.log_softmax(gate_logits, dim=-1)

        expert_logits = []
        for br in self.experts:
            he = self._run_layers(br.layers, h_trunk, attn4d, swm, pos)
            he = br.final_norm(he)
            if self.config.classifier_pooling == "cls":
                pooled = he[:, 0]
            else:  # "mean" — same expression as ModernBertForSequenceClassification
                pooled = (he * attention_mask.unsqueeze(-1)).sum(dim=1) \
                    / attention_mask.sum(dim=1, keepdim=True)
            expert_logits.append(br.classifier(br.drop(br.head(pooled))))
        return gate_logprobs, torch.stack(expert_logits, dim=1)  # (B,E,C)

    @staticmethod
    def blend_logprobs(gate_logprobs, expert_logits):
        """log( sum_e w_e * softmax(logits_e) ), fp32, via logsumexp for stability."""
        logp_e = F.log_softmax(expert_logits.float(), dim=-1)        # (B,E,C)
        return torch.logsumexp(gate_logprobs.unsqueeze(-1) + logp_e, dim=1)

    def forward(self, input_ids=None, attention_mask=None, gate_feats=None,
                labels=None, **kwargs):
        gate_logprobs, expert_logits = self._encode(input_ids, attention_mask,
                                                    gate_feats)
        blend = self.blend_logprobs(gate_logprobs, expert_logits)    # (B,C) logprobs

        loss = None
        if labels is not None:
            eps, C = self.label_smoothing, self.num_labels
            nll = -blend.gather(1, labels.unsqueeze(1)).squeeze(1)
            # torch's label_smoothing target: (1-eps)*onehot + eps/C
            loss = ((1.0 - eps) * nll - eps * blend.mean(dim=-1)).mean()

            w = gate_logprobs.exp()                                  # (B,E)
            # Switch aux (HF mixtral load_balancing_loss_func, top_k=1, per-seq):
            f = F.one_hot(w.argmax(dim=-1), self.n_experts).float().mean(dim=0)
            P = w.mean(dim=0)
            balance = self.n_experts * (f * P).sum()
            # usage-entropy floor on the BATCH-MEAN gate: blocks global collapse,
            # leaves per-row decisiveness (= specialization) unpenalized
            H_usage = -(P.clamp_min(1e-9) * P.clamp_min(1e-9).log()).sum()
            ent_pen = F.relu(self.entropy_tau * torch.log(
                torch.tensor(float(self.n_experts), device=P.device)) - H_usage)
            loss = loss + self.balance_coef * balance + self.entropy_coef * ent_pen
            # running diagnostics for the epoch logger (detached, cheap)
            self._last_aux = {"balance": float(balance.detach()),
                              "usage_H": float(H_usage.detach()),
                              "usage": [round(float(x), 4) for x in P.detach()]}

        return {"loss": loss, "logits": blend} if loss is not None \
            else {"logits": blend}

    @torch.no_grad()
    def predict_parts(self, input_ids, attention_mask, gate_feats=None):
        """Eval helper: per-expert log-probs + gate weights (for blend vs
        uniform-null vs solo read-outs and gate diagnostics)."""
        gate_logprobs, expert_logits = self._encode(input_ids, attention_mask,
                                                    gate_feats)
        return gate_logprobs.exp(), F.log_softmax(expert_logits.float(), dim=-1)

    # ---- persistence (custom: the module is not a PreTrainedModel) ----
    def save_moe(self, run_dir, base_model_name, save_dtype=torch.float16,
                 extra=None):
        os.makedirs(run_dir, exist_ok=True)
        state = {k: v.to(save_dtype) if v.is_floating_point() else v
                 for k, v in self.state_dict().items()}
        torch.save(state, os.path.join(run_dir, MOE_WEIGHTS_NAME))
        meta = {"base_model": base_model_name, "trunk_k": self.trunk_k,
                "n_experts": self.n_experts,
                "gate_hidden": self.gate.mlp[0].out_features,
                "num_labels": self.num_labels,
                "label_smoothing": self.label_smoothing,
                "balance_coef": self.balance_coef,
                "entropy_coef": self.entropy_coef,
                "entropy_tau": self.entropy_tau,
                "id2label": {int(k): v for k, v in
                             dict(self.config.id2label).items()}}
        meta.update(extra or {})
        with open(os.path.join(run_dir, MOE_CONFIG_NAME), "w") as fh:
            json.dump(meta, fh, indent=2)


def load_moe(run_dir, device="cpu", torch_dtype=torch.float32):
    """Rebuild a GraniteMoE from save_moe() output. Uses AutoConfig +
    from_config (random skeleton) then loads the full state_dict — no base
    weights download needed (offline/packaging-safe once the config is cached
    alongside; the config is fetched from base_model unless a config.json sits
    in run_dir)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification
    with open(os.path.join(run_dir, MOE_CONFIG_NAME)) as fh:
        meta = json.load(fh)
    local_cfg = os.path.join(run_dir, "config.json")
    cfg_src = run_dir if os.path.exists(local_cfg) else meta["base_model"]
    cfg = AutoConfig.from_pretrained(cfg_src, num_labels=meta["num_labels"],
                                     id2label={int(k): v for k, v in
                                               meta["id2label"].items()})
    cfg._attn_implementation = "sdpa"
    base = AutoModelForSequenceClassification.from_config(cfg)
    model = GraniteMoE(base, trunk_k=meta["trunk_k"],
                       n_experts=meta["n_experts"],
                       gate_hidden=meta["gate_hidden"],
                       label_smoothing=meta["label_smoothing"],
                       balance_coef=meta["balance_coef"],
                       entropy_coef=meta["entropy_coef"],
                       entropy_tau=meta["entropy_tau"])
    state = torch.load(os.path.join(run_dir, MOE_WEIGHTS_NAME),
                       map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.to(torch_dtype) if v.is_floating_point() else v
                           for k, v in state.items()})
    return model.to(device)
