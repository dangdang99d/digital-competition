"""Fine-tune a backbone with a single linear classification head for the 14-class
next-action task. Metric: Macro-F1 (competition metric).

This run: FULL fine-tuning — all backbone weights + the standard single linear head
are trained (AutoModelForSequenceClassification's built-in head is one linear layer).
Higher ceiling than LoRA but more VRAM/time; uses a small LR (2e-5) and gradient
checkpointing. (The LoRA variant lives on the probe-improvements branch.)

Usage:
  python -m src.finetune --model Qwen/Qwen3-Embedding-0.6B
  python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 --epochs 3
Outputs: output/ft_<model>/ (best checkpoint + metrics), appended to output/ft_results.csv.
"""
import argparse
import csv as _csv
import json
import os

# reduce CUDA fragmentation OOMs (must be set before torch initializes CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import (ALL_CLASSES, CLASS_TO_ID, GROUP_ID, SERIALIZE_VARIANTS,
                      build_texts, load_samples, serialize, split_indices)


def build_dataset(tok, texts, labels, max_len, desc="tokenizing", teacher=None,
                  weights=None):
    """Tokenize once; return a torch Dataset yielding input_ids/attention_mask/labels.

    Tokenizes in chunks with a tqdm bar (works in a terminal and prints periodic
    updates to an sbatch log file).
    teacher: optional (N, num_classes) float array of teacher logits (distillation);
    added to each item as "teacher_logits".
    weights: optional (N,) float array of per-sample loss weights ("weight" key).
    """
    from tqdm.auto import tqdm

    enc = {"input_ids": [], "attention_mask": []}
    chunk = 1000
    for i in tqdm(range(0, len(texts), chunk), desc=desc, unit="k-rows",
                  mininterval=5.0):  # mininterval keeps sbatch logs sparse
        e = tok(texts[i:i + chunk], truncation=True, max_length=max_len, padding=False)
        enc["input_ids"].extend(e["input_ids"])
        enc["attention_mask"].extend(e["attention_mask"])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            item = {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": int(labels[i]),
            }
            if teacher is not None:
                item["teacher_logits"] = teacher[i]
            if weights is not None:
                item["weight"] = float(weights[i])
            return item

    return DS()


def build_dynamic_dataset(tok, samples_sub, labels, max_len, max_hist, hist_dropout, seed,
                          variant="v1"):
    """Train dataset that serializes + tokenizes per __getitem__, so each epoch sees a
    FRESH random history-event drop (static pre-tokenization would corrupt once).
    Per-sample tokenization is ~1ms — negligible next to a full-FT train step."""
    rng = np.random.default_rng(seed)
    var_kw = SERIALIZE_VARIANTS[variant]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            text = serialize(samples_sub[i], max_hist=max_hist,
                             hist_dropout=hist_dropout, rng=rng, **var_kw)
            e = tok(text, truncation=True, max_length=max_len, padding=False)
            return {
                "input_ids": e["input_ids"],
                "attention_mask": e["attention_mask"],
                "labels": int(labels[i]),
            }

    return DS()


def make_compute_metrics(n_classes):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        mf1 = f1_score(labels, preds, labels=list(range(n_classes)),
                       average="macro", zero_division=0)
        return {"macro_f1": mf1}
    return compute_metrics


def _macro_f1(logits, labels, bias):
    preds = np.argmax(logits + bias, axis=1)
    return f1_score(labels, preds, labels=list(range(logits.shape[1])),
                    average="macro", zero_division=0)


def calibrate_logit_bias(logits, labels, rounds=50, grid=None):
    """Post-hoc per-class logit bias tuned on val to maximize Macro-F1 (73.07 trick).

    Coordinate ascent: for each class, try a grid of additive biases and keep the
    value that improves val Macro-F1. Returns (bias_vector, base_f1, tuned_f1).
    rounds=50/step .02: best honest performer in analysis/calibration_methods.py
    (2-fold: 0.7385 vs 0.7376 current, vs 0.7340 for overfit-prone matrix scaling).
    Converges early via the no-improvement break, so cost stays ~1-2 min.
    """
    if grid is None:
        grid = np.round(np.arange(-2.0, 2.001, 0.02), 3)
    n = logits.shape[1]
    bias = np.zeros(n, dtype=np.float32)
    base = _macro_f1(logits, labels, bias)
    best = base
    for _ in range(rounds):
        improved = False
        for c in range(n):
            cur = bias[c]
            best_b, best_f1 = cur, best
            for b in grid:
                bias[c] = b
                f1 = _macro_f1(logits, labels, bias)
                if f1 > best_f1:
                    best_f1, best_b = f1, b
            bias[c] = best_b
            if best_f1 > best:
                best, improved = best_f1, True
        if not improved:
            break
    return bias, base, best


import re as _re

_LAYER_RE = _re.compile(r"\.layers?\.(\d+)\.")


def _layer_depth(model):
    """Number of encoder layers, inferred from parameter names."""
    idxs = [int(m.group(1)) for n, _ in model.named_parameters() if (m := _LAYER_RE.search(n))]
    return max(idxs) + 1 if idxs else 0


def build_llrd_optimizer(model, base_lr, decay, optim_name, weight_decay):
    """Optimizer with layer-wise LR decay: layer i gets base_lr * decay^(depth-1-i);
    embeddings get one step lower; head/pooler get full base_lr."""
    depth = _layer_depth(model)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = _LAYER_RE.search(n)
        if m:
            lr = base_lr * decay ** (depth - 1 - int(m.group(1)))
        elif "embed" in n:
            lr = base_lr * decay ** depth
        else:                       # head / pooler / final norms
            lr = base_lr
        groups.setdefault(round(lr, 12), []).append(p)
    param_groups = [{"params": ps, "lr": lr} for lr, ps in groups.items()]
    logger.info(f"LLRD: {len(param_groups)} LR groups, "
                f"min={min(g['lr'] for g in param_groups):.2e} max={base_lr:.2e}")
    if optim_name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(param_groups, lr=base_lr, scale_parameter=False,
                         relative_step=False, warmup_init=False, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def reinit_top_layers(model, n):
    """Re-initialize the top n encoder layers (fresh start for task-specific tops)."""
    layer_lists = [mod for name, mod in model.named_modules()
                   if name.endswith(("encoder.layer", "encoder.layers", "model.layers"))]
    assert layer_lists, "could not locate the encoder layer list for --reinit_layers"
    layers = layer_lists[0]
    for layer in list(layers)[-n:]:
        layer.apply(model._init_weights)
    logger.info(f"re-initialized top {n} encoder layers")


def prune_layers(model, keep):
    """Structured depth pruning: keep `keep` evenly-spaced encoder layers (always
    including the first and last), drop the rest. config.num_hidden_layers is updated
    so the pruned checkpoint reloads with from_pretrained. Pair with --init_from to
    prune a trained model and fine-tune to recover."""
    hit = next(((name, mod) for name, mod in model.named_modules()
                if name.endswith(("encoder.layer", "encoder.layers", "model.layers"))), None)
    assert hit, "could not locate the encoder layer list for --keep_layers"
    list_name, layers = hit
    depth = len(layers)
    assert keep < depth, f"--keep_layers {keep} >= model depth {depth}"
    idx = sorted(set(np.round(np.linspace(0, depth - 1, keep)).astype(int).tolist()))
    parent = model.get_submodule(list_name.rsplit(".", 1)[0])
    setattr(parent, list_name.rsplit(".", 1)[1],
            torch.nn.ModuleList([layers[i] for i in idx]))
    model.config.num_hidden_layers = len(idx)
    logger.info(f"pruned encoder depth {depth} -> {len(idx)} (kept layers {idx})")


class ExtrasCollator:
    """Wrap DataCollatorWithPadding: non-text keys (teacher_logits, weight) can't go
    through tokenizer.pad, so pop them, pad the rest, and re-attach as tensors."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        teacher = [f.pop("teacher_logits", None) for f in features]
        weight = [f.pop("weight", None) for f in features]
        batch = self.base(features)
        if teacher[0] is not None:
            batch["teacher_logits"] = torch.tensor(np.stack(teacher), dtype=torch.float32)
        if weight[0] is not None:
            batch["weight"] = torch.tensor(weight, dtype=torch.float32)
        return batch


def prune_ffn(model, keep_ratio):
    """Structured width pruning of every FFN: keep the top keep_ratio intermediate
    neurons by |W_in row| * |W_out col| (throughput proxy), slice both matrices.
    config.intermediate_size is updated so the checkpoint reloads cleanly."""
    import torch.nn as nn

    k = None
    n_pruned = 0
    for name, layer in model.named_modules():
        if not (hasattr(layer, "intermediate") and hasattr(layer, "output")):
            continue
        if not (hasattr(layer.intermediate, "dense") and hasattr(layer.output, "dense")):
            continue
        w_in, w_out = layer.intermediate.dense, layer.output.dense
        inter = w_in.out_features
        k = max(1, int(round(inter * keep_ratio)))
        score = w_in.weight.data.norm(dim=1) * w_out.weight.data.norm(dim=0)
        idx = torch.topk(score, k).indices.sort().values
        new_in = nn.Linear(w_in.in_features, k)
        new_in.weight.data = w_in.weight.data[idx].clone()
        new_in.bias.data = w_in.bias.data[idx].clone()
        new_out = nn.Linear(k, w_out.out_features)
        new_out.weight.data = w_out.weight.data[:, idx].clone()
        new_out.bias.data = w_out.bias.data.clone()
        layer.intermediate.dense, layer.output.dense = new_in, new_out
        n_pruned += 1
    assert n_pruned, "no intermediate/output FFN pairs found for --ffn_keep"
    model.config.intermediate_size = k
    logger.info(f"FFN width pruned in {n_pruned} layers: {inter} -> {k} neurons")


def prune_attn_heads(model, keep_ratio):
    """Drop the least important attention heads per layer (|W_v head| * |W_o head|
    proxy) via HF's model.prune_heads; config.pruned_heads records the surgery so
    from_pretrained reloads correctly. Call AFTER prune_layers (indices re-read)."""
    n_heads = model.config.num_attention_heads
    k = max(1, int(round(n_heads * keep_ratio)))
    to_prune = {}
    for name, mod in model.named_modules():
        if not name.endswith("attention.self"):
            continue
        layer_idx = int(name.split(".layer.")[1].split(".")[0])
        hd = mod.attention_head_size
        out = model.get_submodule(name.rsplit(".self", 1)[0] + ".output.dense")
        scores = [(mod.value.weight.data[h * hd:(h + 1) * hd].norm()
                   * out.weight.data[:, h * hd:(h + 1) * hd].norm()).item()
                  for h in range(n_heads)]
        to_prune[layer_idx] = sorted(range(n_heads), key=scores.__getitem__)[:n_heads - k]
    assert to_prune, "no attention.self modules found for --heads_keep"
    model.prune_heads(to_prune)
    logger.info(f"attention heads pruned in {len(to_prune)} layers: {n_heads} -> {k}")


def replace_head(model, n_layers, act="gelu"):
    """Swap the classification head for an n_layers MLP (act nonlinearity, dropout
    between).

    Handles both head styles:
      - Qwen3-style `model.score`: applied per-token, the model selects the last
        non-pad position afterwards -> plain Sequential works token-wise.
      - XLM-R-style `model.classifier`: called with the full sequence output and
        pools [CLS] internally -> wrapper reproduces the features[:, 0] selection.
        (NOTE: the stock XLM-R head is already dense+tanh+out_proj, i.e. ~2 layers;
        n_layers=3 is the genuinely deeper variant there.)
    """
    import torch.nn as nn

    hidden = model.config.hidden_size
    p = getattr(model.config, "classifier_dropout", None) or 0.1
    num_labels = model.config.num_labels

    act_layer = {"gelu": nn.GELU, "tanh": nn.Tanh}[act]

    def mlp():
        layers, in_dim = [], hidden
        for _ in range(n_layers - 1):
            layers += [nn.Dropout(p), nn.Linear(in_dim, hidden), act_layer()]
            in_dim = hidden
        layers += [nn.Dropout(p), nn.Linear(in_dim, num_labels)]
        return nn.Sequential(*layers)

    if hasattr(model, "score"):
        model.score = mlp()
        new = model.score
    elif hasattr(model, "classifier"):
        class CLSHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = mlp()

            def forward(self, features, **kwargs):
                x = features[:, 0, :] if features.dim() == 3 else features
                return self.mlp(x)

        model.classifier = CLSHead()
        new = model.classifier
    else:
        raise AssertionError("no .score or .classifier head found for --head_layers")
    new.apply(model._init_weights)
    # recorded so downstream loaders can reconstruct: plain from_pretrained rebuilds
    # the STOCK head and silently drops classifier.mlp.* weights. To reload a custom-
    # head checkpoint: build the base model, call replace_head(m, **config.custom_head
    # values), then load the safetensors state dict.
    model.config.custom_head = {"layers": n_layers, "act": act}
    logger.info(f"replaced classification head with {n_layers}-layer {act} MLP (dropout {p})")


def make_distill_trainer(trainer_cls, alpha, temperature):
    import torch.nn.functional as F

    class _DistillTrainer(trainer_cls):
        """loss = (1-alpha)*CE(student, y) + alpha*T^2*KL(teacher_T || student_T).
        Eval batches carry no teacher_logits -> plain CE there."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            teacher_logits = inputs.pop("teacher_logits", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if teacher_logits is not None:
                kd = F.kl_div(
                    F.log_softmax(outputs.logits.float() / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature ** 2
                loss = (1.0 - alpha) * loss + alpha * kd
            return (loss, outputs) if return_outputs else loss

    return _DistillTrainer


def make_aux_trainer(base_cls, rdrop=0.0, supcon=None):
    """Trainer with optional boundary-sharpening losses (all train-only, eval is
    plain CE):

      per-sample weights  — "weight" batch key (from --hard_boundary mining):
                            loss = sum(w_i * CE_i) / sum(w_i)
      rdrop > 0           — second forward with fresh dropout masks;
                            loss = mean CE of both + rdrop * symmetric KL
      supcon dict         — SupCon aux on the pooled embedding via model.supcon_proj
                            (lam, tau, group_w, queue): cross-batch memory queue
                            supplies positives, so no batch-composition surgery;
                            group_w > 1 upweights SAME-GROUP negatives (read vs grep)
                            in the denominator — the boundaries we actually lose on.
    """
    import torch.nn.functional as F

    class _AuxTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weight = inputs.pop("weight", None)
            # hidden states only during training — in eval they'd leak into the
            # Trainer's prediction tuple and break compute_metrics
            need_h = supcon is not None and model.training
            outputs = model(**inputs, output_hidden_states=need_h)
            loss = outputs.loss
            if not model.training:               # eval path: plain CE
                return (loss, outputs) if return_outputs else loss
            labels = inputs["labels"]
            if weight is not None:
                ce = F.cross_entropy(outputs.logits.float(), labels, reduction="none")
                loss = (ce * weight).sum() / weight.sum()
            if rdrop:
                out2 = model(**inputs)           # fresh dropout masks
                lp1 = F.log_softmax(outputs.logits.float(), dim=-1)
                lp2 = F.log_softmax(out2.logits.float(), dim=-1)
                kl = 0.5 * (F.kl_div(lp1, lp2, log_target=True, reduction="batchmean")
                            + F.kl_div(lp2, lp1, log_target=True, reduction="batchmean"))
                loss = 0.5 * (loss + out2.loss) + rdrop * kl
            if supcon is not None:
                loss = loss + supcon["lam"] * self._supcon_loss(model, outputs, inputs)
            return (loss, outputs) if return_outputs else loss

        def _supcon_loss(self, model, outputs, inputs):
            h = outputs.hidden_states[-1]
            if hasattr(model, "score"):          # Qwen-style: last non-pad token
                last = inputs["attention_mask"].sum(1) - 1
                pooled = h[torch.arange(h.shape[0], device=h.device), last]
            else:                                # XLM-R-style: [CLS]
                pooled = h[:, 0]
            with torch.autocast(pooled.device.type, enabled=False):  # fp32 for stability
                z = F.normalize(model.supcon_proj(pooled.float()), dim=-1)
            y = inputs["labels"]
            if not hasattr(self, "_xbm_z"):      # lazy cross-batch memory init
                self._xbm_z = z.new_zeros((0, z.shape[1]))
                self._xbm_y = y.new_zeros((0,))
                self._gid = torch.tensor(GROUP_ID, device=y.device)
            bank_z = torch.cat([z.detach(), self._xbm_z])
            bank_y = torch.cat([y, self._xbm_y])
            sim = z @ bank_z.T / supcon["tau"]                       # (B, B+Q)
            b = z.shape[0]
            self_mask = torch.zeros_like(sim, dtype=torch.bool)
            self_mask[:, :b] = torch.eye(b, dtype=torch.bool, device=sim.device)
            pos = (bank_y[None, :] == y[:, None]) & ~self_mask
            w = torch.ones_like(sim)
            same_group = self._gid[bank_y][None, :] == self._gid[y][:, None]
            w[same_group & ~pos] = supcon["group_w"]                 # hard negatives
            w[self_mask] = 0.0
            smax = sim.max(1, keepdim=True).values
            denom = (w * torch.exp(sim - smax)).sum(1, keepdim=True)
            log_prob = sim - smax - torch.log(denom + 1e-12)
            n_pos = pos.sum(1)
            per_anchor = -(log_prob * pos).sum(1) / n_pos.clamp(min=1)
            loss = per_anchor[n_pos > 0].mean() if (n_pos > 0).any() else sim.sum() * 0.0
            # FIFO queue update (detached)
            self._xbm_z = torch.cat([z.detach(), self._xbm_z])[: supcon["queue"]]
            self._xbm_y = torch.cat([y, self._xbm_y])[: supcon["queue"]]
            return loss

    return _AuxTrainer


def mine_boundary_weights(model, tok, texts_tr, max_len, weight, margin, device):
    """One inference pass over the train split with the --init_from checkpoint:
    samples whose top-2 logits are BOTH in the same confusion group and closer
    than `margin` get loss weight `weight` (they sit on the boundary we lose on);
    everything else keeps 1.0."""
    gid = np.array(GROUP_ID)
    weights = np.ones(len(texts_tr), dtype=np.float32)
    order = sorted(range(len(texts_tr)), key=lambda i: -len(texts_tr[i]))  # pad less
    model.to(device).eval()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad():
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            enc = tok([texts_tr[i] for i in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                lg = model(**enc).logits.float().cpu().numpy()
            top2 = np.argsort(lg, axis=1)[:, -2:]                    # [second, first]
            marg = np.take_along_axis(lg, top2, 1)
            flag = (gid[top2[:, 0]] == gid[top2[:, 1]]) & ((marg[:, 1] - marg[:, 0]) < margin)
            for j, i in enumerate(idx):
                if flag[j]:
                    weights[i] = weight
    frac = float((weights > 1).mean())
    logger.info(f"boundary mining: {frac:.1%} of train samples flagged "
                f"(weight {weight}, margin {margin})")
    return weights


def add_action_tokens(tok, model):
    """Add the 14 action names + serialization markers as ATOMIC tokens.

    Without this, bge-m3's sentencepiece splits every action name into 3-7 pieces
    ('lint_or_typecheck' -> 7). Each new token gets ONE embedding row, initialized
    to the MEAN of the piece embeddings it replaces — training starts from the
    compositional representation instead of random. The tokenizer must be saved
    with the run (save_pretrained) and shipped at inference, or the ids shift.
    """
    new_tokens = list(ALL_CLASSES) + ["ACTION", "USER:", "PROMPT:"]
    piece_ids = {t: tok(t, add_special_tokens=False)["input_ids"] for t in new_tokens}
    n_added = tok.add_tokens(new_tokens)
    # when continuing from a checkpoint that ALREADY has the extra rows
    # (--init_from a spectok run), resize is a no-op and the trained embeddings
    # must NOT be re-initialized — only init rows that did not exist before.
    old_rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    inited = 0
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        for t in new_tokens:
            new_id = tok.convert_tokens_to_ids(t)
            if new_id >= old_rows:
                w[new_id] = w[piece_ids[t]].mean(dim=0)
                inited += 1
    logger.info(f"atomic action/marker tokens: {n_added} added to tokenizer "
                f"(vocab -> {len(tok)}), {inited} embedding rows mean-of-pieces "
                f"initialized ({n_added - inited} pre-trained rows kept)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="context", choices=["context", "prompt"])
    ap.add_argument("--max_hist", type=int, default=0,
                    help="cap history to last N events; 0 = full history (no cap)")
    ap.add_argument("--serialize", default="v1", choices=sorted(SERIALIZE_VARIANTS),
                    help="input serialization variant (see data.SERIALIZE_VARIANTS); "
                         "v1 = the hist0-baseline format")
    ap.add_argument("--special_tokens", action="store_true",
                    help="add the 14 action names + ACTION/USER:/PROMPT: markers as "
                         "single tokens (mean-of-pieces embedding init); the run-dir "
                         "tokenizer must then be used at inference")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)          # full-FT needs a small LR
    ap.add_argument("--batch_size", type=int, default=4)       # full-FT is VRAM-heavy (~11GB GPU)
    ap.add_argument("--grad_accum", type=int, default=4)       # effective batch 16
    ap.add_argument("--optim", default="adamw_torch",
                    help="optimizer. adamw_torch (default, best quality) or sgd "
                         "(zero optimizer state -> fits bigger models like Qwen3 full-FT). "
                         "SGD usually needs a higher --lr.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--results_name", default="ft_results.csv",
                    help="results CSV filename; give each array task a unique one to "
                         "avoid concurrent-append races")
    ap.add_argument("--limit", type=int, default=0, help="cap train+val size (0=all); for quick tests")
    ap.add_argument("--init_from", default="",
                    help="path to a fine-tuned checkpoint dir to CONTINUE training from "
                         "(weights load from here; --model still names the tokenizer/base)")
    ap.add_argument("--full_data", action="store_true",
                    help="fold 75%% of the val split into training; the remaining 25%% "
                         "(stratified) stays held out for eval + calibration. Safe to combine "
                         "with --init_from: the checkpoint only ever saw the train split")
    ap.add_argument("--tag", default="",
                    help="suffix for the run dir (ft_<model>_<tag>) so reruns don't clobber "
                         "earlier checkpoints of the same model")
    ap.add_argument("--keep_checkpoints", type=int, default=1,
                    help="how many epoch checkpoints to keep (save_total_limit). Set >1 to "
                         "enable post-hoc SWA averaging (analysis/swa_average.py); those "
                         "checkpoints are saved model-only (no optimizer state) to spare disk")
    ap.add_argument("--llrd", type=float, default=0.0,
                    help="layer-wise LR decay factor (e.g. 0.9): layer i gets lr*decay^(depth-i); "
                         "embeddings lowest, head full lr. 0 = off (uniform lr)")
    ap.add_argument("--reinit_layers", type=int, default=0,
                    help="re-initialize the top N encoder layers before training (retrieval-"
                         "pretrained tops may transfer worse than a fresh start)")
    ap.add_argument("--hist_dropout", type=float, default=0.0,
                    help="training-time augmentation: drop each history event with this "
                         "probability, re-drawn every epoch (val is never dropped)")
    ap.add_argument("--keep_layers", type=int, default=0,
                    help="depth-prune the encoder to N evenly-spaced layers before training "
                         "(0 = off). Pair with --init_from for prune-then-recover")
    ap.add_argument("--distill_from", default="",
                    help="path to a .npz with key 'logits' of shape (n_samples, 14) in "
                         "load_samples order — teacher logits for distillation "
                         "(dump with src/dump_logits.py)")
    ap.add_argument("--distill_alpha", type=float, default=0.5,
                    help="weight of the KD term: loss = (1-a)*CE + a*T^2*KL")
    ap.add_argument("--distill_T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--head_layers", type=int, default=0,
                    help="replace the classification head with an N-layer GELU MLP "
                         "(0 = keep the model's stock head)")
    ap.add_argument("--head_act", default="gelu", choices=["gelu", "tanh"],
                    help="nonlinearity for the --head_layers MLP")
    ap.add_argument("--ffn_keep", type=float, default=1.0,
                    help="structured width pruning: keep this fraction of FFN "
                         "intermediate neurons per layer (1.0 = off)")
    ap.add_argument("--heads_keep", type=float, default=1.0,
                    help="structured width pruning: keep this fraction of attention "
                         "heads per layer (1.0 = off)")
    ap.add_argument("--rdrop", type=float, default=0.0,
                    help="R-Drop: second forward pass with fresh dropout, symmetric-KL "
                         "consistency penalty with this weight (0 = off). Noise-"
                         "compatible boundary sharpening; ~doubles train compute")
    ap.add_argument("--supcon", type=float, default=0.0,
                    help="weight of a supervised-contrastive auxiliary loss on the "
                         "pooled embedding (0 = off). Uses a cross-batch memory queue "
                         "for positives; projection head is train-only (not shipped)")
    ap.add_argument("--supcon_tau", type=float, default=0.1, help="SupCon temperature")
    ap.add_argument("--supcon_group_w", type=float, default=2.0,
                    help="upweight same-group negatives in the SupCon denominator "
                         "(read-vs-grep style hard negatives); 1.0 = uniform")
    ap.add_argument("--supcon_queue", type=int, default=4096,
                    help="cross-batch memory size (embeddings kept as extra pos/neg)")
    ap.add_argument("--hard_boundary", type=float, default=0.0,
                    help="hard-example mining (needs --init_from): upweight train "
                         "samples whose top-2 checkpoint logits are same-group and "
                         "closer than --boundary_margin to this loss weight (0 = off)")
    ap.add_argument("--boundary_margin", type=float, default=2.0,
                    help="logit-margin threshold for --hard_boundary mining")
    ap.add_argument("--pair", default="",
                    help="binary specialist mode: 'classA,classB' — train only on "
                         "samples of these two classes with a 2-class head (init the "
                         "encoder from --init_from). For margin-gated deferral")
    args = ap.parse_args()

    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  model={args.model}  method=full-finetune+linear-head")

    # ---- data ----
    samples, y = load_samples(args.data_dir)
    max_hist = args.max_hist or None            # 0 -> None (full history)
    texts = build_texts(samples, input_mode=args.input, max_hist=max_hist,
                        variant=args.serialize)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    if args.full_data:
        # grow training with 75% of the old val; eval/calibrate on the untouched 25%.
        # Same seed as split_indices -> an --init_from checkpoint trained on `tr`
        # has never seen ANY of the surviving eval samples.
        from sklearn.model_selection import train_test_split
        va_train, va_eval = train_test_split(
            va, test_size=0.25, stratify=y_ids[va], random_state=args.seed)
        tr = np.concatenate([tr, va_train])
        va = va_eval
    classes = ALL_CLASSES
    if args.pair:
        classes = args.pair.split(",")
        assert len(classes) == 2 and all(c in CLASS_TO_ID for c in classes), \
            f"--pair must name two of {ALL_CLASSES}"
        # filter AFTER the split, keeping absolute indices: the specialist's train
        # set stays inside the 14-class model's train split, so composing the two
        # on the main val split later is uncontaminated
        pair_ids = {CLASS_TO_ID[c] for c in classes}
        y_ids = np.array([classes.index(a) if CLASS_TO_ID[a] in pair_ids else -1
                          for a in y])
        tr = tr[y_ids[tr] >= 0]
        va = va[y_ids[va] >= 0]
        logger.info(f"pair specialist {classes}: filtered to {len(tr)} train / {len(va)} val")
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    logger.info(f"samples={len(texts)}  train={len(tr)}  val={len(va)}  max_hist={max_hist}  "
                f"full_data={args.full_data}")

    # ---- tokenizer + model + single linear head ----
    # trust_remote_code: some backbones (e.g. gte's model_type "new") ship custom
    # modeling code and won't load without it.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.init_from:
        logger.info(f"continuing from fine-tuned checkpoint: {args.init_from}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init_from or args.model, num_labels=len(classes),
        torch_dtype=torch.float32,   # fp32 for stable classifier training
        trust_remote_code=True,
        # real action names so config.id2label maps ids -> actions (not LABEL_0...);
        # needed for logit_bias.json keys and for inference to emit action strings.
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)},
        # some backbones ship a pretrained head (e.g. gte has a 1-logit head);
        # discard it and init a fresh 14-class head for our task.
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    if args.special_tokens:
        add_action_tokens(tok, model)
    if args.reinit_layers:
        reinit_top_layers(model, args.reinit_layers)
    if args.keep_layers:
        prune_layers(model, args.keep_layers)
    if args.ffn_keep < 1.0:
        prune_ffn(model, args.ffn_keep)
    if args.heads_keep < 1.0:
        prune_attn_heads(model, args.heads_keep)   # after prune_layers: fresh indices
    if args.head_layers:
        replace_head(model, args.head_layers, args.head_act)

    # ---- full fine-tuning: ALL backbone weights + head are trainable ----
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {n_trainable:,} / {n_total:,} (100% — full fine-tune)")

    aux_on = bool(args.rdrop or args.supcon or args.hard_boundary)
    if aux_on:
        assert not args.distill_from, \
            "--distill_from is not combinable with --rdrop/--supcon/--hard_boundary"
    assert not (args.rdrop and args.hard_boundary), \
        "--rdrop + --hard_boundary: the KL term would mix weighted/unweighted CE"
    if args.supcon:
        assert not args.pair, "--supcon group weighting assumes the 14-class space"
        import torch.nn as nn
        h = model.config.hidden_size
        # train-only projection head (SupCon standard); saved with checkpoints but
        # ignored by from_pretrained at inference — nothing ships
        model.supcon_proj = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 128))
        logger.info(f"SupCon aux: lam={args.supcon} tau={args.supcon_tau} "
                    f"group_w={args.supcon_group_w} queue={args.supcon_queue}")
    weights_tr = None
    if args.hard_boundary:
        assert args.init_from, "--hard_boundary mines boundaries with the --init_from checkpoint"
        assert not args.hist_dropout, "--hard_boundary needs static per-sample weights"
        weights_tr = mine_boundary_weights(
            model, tok, [texts[i] for i in tr], args.max_len,
            args.hard_boundary, args.boundary_margin, device)

    teacher = None
    if args.distill_from:
        teacher = np.load(args.distill_from)["logits"].astype(np.float32)
        assert len(teacher) == len(texts), \
            f"teacher logits rows ({len(teacher)}) != samples ({len(texts)})"
        logger.info(f"distilling from {args.distill_from} "
                    f"(alpha={args.distill_alpha}, T={args.distill_T})")
    if args.hist_dropout:
        assert not args.distill_from, "--hist_dropout + --distill_from not supported together"
        logger.info(f"history dropout p={args.hist_dropout} (fresh draw per epoch)")
        train_ds = build_dynamic_dataset(tok, [samples[i] for i in tr], y_ids[tr],
                                         args.max_len, max_hist, args.hist_dropout, args.seed,
                                         variant=args.serialize)
    else:
        train_ds = build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len,
                                 teacher=teacher[tr] if teacher is not None else None,
                                 weights=weights_tr)
    val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len)
    collator = DataCollatorWithPadding(tok)
    if teacher is not None or weights_tr is not None:
        collator = ExtrasCollator(collator)

    # plain cross-entropy (Trainer's default). No class weighting — the 73.07 model
    # handles imbalance via post-hoc logit-bias calibration instead.
    safe = args.model.replace("/", "__") + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(args.out_dir, f"ft_{safe}")
    # persist the tokenizer with the run: with --special_tokens the vocab differs
    # from the hub's, and inference MUST load this copy or every id past the old
    # vocab end is wrong. Saved unconditionally — harmless for stock runs.
    tok.save_pretrained(run_dir)
    # bf16 where the GPU supports it (Ampere+: 3090/4090). fp16 NaN-diverged
    # Qwen3-0.6B on pat (loss -> 0.0, grad_norm NaN mid-epoch-1) while the same
    # recipe was fine for granite — LLM-style backbones overflow fp16's range.
    # 2080 Ti (Turing) has no bf16 -> falls back to fp16 as before.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    logger.info(f"precision: {'bf16' if use_bf16 else 'fp16' if device == 'cuda' else 'fp32'}")
    targs = TrainingArguments(
        output_dir=run_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        optim=args.optim,                              # sgd for big models (zero optimizer state)
        warmup_ratio=0.05, weight_decay=0.01,
        logging_strategy="steps", logging_steps=50,   # periodic {loss,epoch} log lines
        disable_tqdm=False,                            # keep the bar; tqdm.auto is log-safe
        gradient_checkpointing=True,                   # trade compute for VRAM (full-FT is heavy)
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=args.keep_checkpoints,
        save_only_model=(args.keep_checkpoints > 1),   # SWA needs weights only; saves ~2x disk
        bf16=use_bf16, fp16=(device == "cuda" and not use_bf16),
        report_to="none", seed=args.seed,
    )
    optimizers = (None, None)
    if args.llrd:
        optimizers = (build_llrd_optimizer(model, args.lr, args.llrd,
                                           args.optim, targs.weight_decay), None)
    trainer_cls = Trainer
    if teacher is not None:
        trainer_cls = make_distill_trainer(Trainer, args.distill_alpha, args.distill_T)
    if aux_on:
        supcon_cfg = ({"lam": args.supcon, "tau": args.supcon_tau,
                       "group_w": args.supcon_group_w, "queue": args.supcon_queue}
                      if args.supcon else None)
        trainer_cls = make_aux_trainer(Trainer, rdrop=args.rdrop, supcon=supcon_cfg)
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=make_compute_metrics(len(classes)),
        optimizers=optimizers,
    )

    trainer.train()
    metrics = trainer.evaluate()
    val_f1 = metrics["eval_macro_f1"]
    logger.success(f"{args.model}: best val Macro-F1 = {val_f1:.4f}")

    # ---- logit-bias calibration on val (73.07 trick) ----
    pred_out = trainer.predict(val_ds)
    val_logits = pred_out.predictions
    val_labels = pred_out.label_ids
    bias, base_f1, tuned_f1 = calibrate_logit_bias(val_logits, val_labels)
    logger.success(f"  calibrated: {base_f1:.4f} -> {tuned_f1:.4f} (+{tuned_f1-base_f1:.4f})")
    # save the bias next to the model checkpoint for inference
    id2label = model.config.id2label
    bias_map = {id2label[i]: float(bias[i]) for i in range(len(classes))}
    with open(os.path.join(run_dir, "logit_bias.json"), "w") as f:
        json.dump({"base_macro_f1": float(base_f1), "tuned_macro_f1": float(tuned_f1),
                   "bias": bias_map}, f, indent=2)

    # ---- record ----
    os.makedirs(args.out_dir, exist_ok=True)
    row = {"model": args.model, "method": "full_ft",
           "head": f"mlp{args.head_layers}_{args.head_act}" if args.head_layers else "linear",
           "epochs": args.epochs, "lr": args.lr, "val_macro_f1": round(float(val_f1), 4),
           "calibrated_macro_f1": round(float(tuned_f1), 4),
           "tag": args.tag, "max_len": args.max_len, "serialize": args.serialize,
           "special_tokens": args.special_tokens,
           "init_from": args.init_from, "full_data": args.full_data,
           "rdrop": args.rdrop, "supcon": args.supcon,
           "hard_boundary": args.hard_boundary, "pair": args.pair}
    out_csv = os.path.join(args.out_dir, args.results_name)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info(f"appended -> {out_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
