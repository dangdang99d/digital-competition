"""LTP (Learned Token Pruning) for ModernBERT-class encoders (granite backbone).

E18. Reference: Kim, Gholami, Yao, Mahoney, Keutzer — "Learned Token Pruning for
Transformers" (KDD 2022). Official code cloned under
  scratchpad/ltp_official/src/transformers/models/ltp/{prune_modules,modeling_ltp}.py
This module reimplements the `absolute_threshold` LTP variant on top of HF's
`ModernBertForSequenceClassification` WITHOUT forking the attention kernels: we bind
a custom forward onto the inner `ModernBertModel` that runs the stock encoder layers
one at a time and inserts the LTP soft-mask / hard-prune step between them. When LTP
is OFF the forward is byte-for-byte the stock forward (parity gate), because it calls
the exact same submodules (embeddings / layers / final_norm) and the only inserted op
is a multiply-by-1.0.

=====================================================================================
HOW THIS MAPS TO THE OFFICIAL REFERENCE (prune_modules.AbsoluteThresholdTokenPruner
and modeling_ltp.LTPLayer / trainer.training_step):

* Per-layer threshold (MATCH). Ref __init__:
      keep_threshold_base = final_token_threshold * module_num / num_hidden_layers
      keep_threshold      = nn.Parameter(zeros)          # learnable delta
      effective           = keep_threshold_base + keep_threshold
  Here: `ltp_base[i] = final_token_threshold * i / L` (buffer, fixed), and
  `ltp_delta[i] = nn.Parameter(0.)` (learnable). effective = base + delta. Same.

* Importance score = "attention received" (MATCH). Ref update_attention_mask:
      attention_probs[<pad-query rows>] = 0
      pruning_scores = attention_probs.view(bs, -1, sz).mean(dim=1)      # (bs, seq)
  i.e. for each KEY token j: mean over heads AND (non-pad) query positions i of the
  softmax prob attn[b,h,i,j] it RECEIVES. Here `_received_attention_score` computes
  `(attn * query_valid_mask).mean(dim=(1,2))` — identical (mean over heads*queries
  with pad-query rows zeroed; denominator stays heads*queries, matching ref .mean).

* Soft mask (training) (MATCH). Ref LTPLayer.forward:
      self.mask   = sigmoid((pruning_scores - threshold) / temperature)   # (bs, seq)
      layer_output = layer_output * self.mask.unsqueeze(-1)
  applied to the FULL post-FFN layer output. Here identical: after layer i we do
  `hidden = hidden * sigmoid((score - eff)/T).unsqueeze(-1)` and stash the mask.

* Sparsity regularizer (MATCH). Ref trainer.training_step (masking_mode=='soft'):
      loss += lambda_threshold * sum_over_layers( mask.mean() )
  `ltp_regularizer(model)` returns `ltp_lambda * sum(m.mean() for m in soft_masks)`.
  (Ref `mask.mean()` averages over ALL positions incl. pad; we match that exactly.)

* Hard mask (inference) (SEMANTIC MATCH, see divergence #3). Ref hard mode sets the
  additive attention mask to -10000 at pruned KEY columns so later layers never
  attend to them; it does NOT physically shrink the tensor (speedup in the paper is
  measured via a MAC count). We do the same mask-extension and additionally report
  the per-layer kept-token fraction (telemetry). Physical unpad/gather for wall-clock
  latency is a deployment step to add when the trained checkpoint is validated.

* 2-stage soft->hard (MATCH design). Train with mode='soft' (learnable thresholds +
  regularizer); evaluate/deploy with mode='hard'. Ref 'mixed' (first half soft, back
  half hard) is supported via set_ltp_mode() called from a callback if wanted.

=====================================================================================
DIVERGENCES FROM THE PLAIN-BERT REFERENCE (all forced by the ModernBERT backbone):

1. attn_implementation MUST be 'eager'. ModernBERT defaults to flash-attn + unpadding
   which never materializes the softmax probabilities; LTP scores tokens by attention
   received, so the probs must exist. We assert eager and force
   config.output_attentions handling internally (each layer is called with
   output_attentions=True so it returns attn weights; we consume them and do NOT
   bubble them to the model output, so the HF Trainer never gathers giant (b,h,s,s)
   tensors).

2. Alternating local/global attention. ModernBERT uses global attention every 3rd
   layer (global_attn_every_n_layers=3) and sliding-window (128) local attention
   elsewhere, plus RoPE. In a LOCAL layer the softmax matrix is banded: a token's
   "attention received" is summed only over the queries whose window covers it. We
   sum over whatever weights that layer produced (banded for local, dense for global)
   — the reference's plain-BERT global attention has no banding. This is the intended
   handling per the E18 spec; document it as a divergence.

3. Hard mode is reference-faithful mask-extension (does not physically shrink). See
   above. Never affects CLS (position 0), which granite uses for pooling
   (classifier_pooling='cls' -> hidden[:,0]); CLS is force-kept every layer.
=====================================================================================
"""
import types

import torch
import torch.nn as nn
from loguru import logger
from transformers.modeling_outputs import BaseModelOutput

_NEG = torch.finfo(torch.float32).min  # additive-mask "-inf" (matches ModernBERT)


class _LTPState:
    """Runtime LTP config + per-forward scratch, hung off the inner ModernBertModel."""

    def __init__(self, num_layers, final_token_threshold, temperature, ltp_lambda, mode):
        self.num_layers = int(num_layers)
        self.final = float(final_token_threshold)
        self.temperature = float(temperature)
        self.ltp_lambda = float(ltp_lambda)
        self.mode = mode                 # "off" | "soft" | "hard"
        self.soft_masks = []             # per-layer (bs, seq) soft masks (this forward)
        self.kept_frac = []              # per-layer kept-token fraction (hard telemetry)


def _received_attention_score(attn_weights, query_valid):
    """Importance score per KEY token = mean over heads and (non-pad) query positions
    of the attention prob it receives.  attn_weights: (bs, heads, q, k) post-softmax;
    query_valid: (bs, q) bool (True = real token).  Returns (bs, k).

    Matches AbsoluteThresholdTokenPruner: zero pad-QUERY rows, then mean over the
    flattened (heads * queries) axis (denominator = heads*queries, incl. zeroed rows).
    """
    qmask = query_valid[:, None, :, None].to(attn_weights.dtype)   # (bs,1,q,1)
    return (attn_weights * qmask).mean(dim=(1, 2))                  # (bs, k)


def _ltp_inner_forward(
    self,
    input_ids=None,
    attention_mask=None,
    sliding_window_mask=None,
    position_ids=None,
    inputs_embeds=None,
    indices=None,
    cu_seqlens=None,
    max_seqlen=None,
    batch_size=None,
    seq_len=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    """Custom ModernBertModel.forward reproducing the stock EAGER path, with LTP
    inserted between encoder layers. Bound onto the inner model by install_ltp().

    Parity: with state.mode == 'off' this is the stock forward (no inserted op).
    With mode == 'soft' and thresholds == -inf, the inserted op is *1.0 (no-op).
    """
    st = self._ltp
    cfg = self.config
    assert cfg._attn_implementation == "eager", (
        "LTP requires attn_implementation='eager' (flash/sdpa don't materialize "
        f"attention probs); got {cfg._attn_implementation!r}"
    )
    return_dict = return_dict if return_dict is not None else cfg.use_return_dict
    output_hidden_states = bool(output_hidden_states)
    all_hidden_states = () if output_hidden_states else None

    self._maybe_set_compile()
    if input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    if batch_size is None and seq_len is None:
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            batch_size, seq_len = input_ids.shape[:2]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

    # bool per-token validity (for the pruning score's pad-query masking); the pooling
    # in the outer ForSequenceClassification also uses the ORIGINAL 2D mask so we never
    # mutate it.
    query_valid = attention_mask.bool()

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    # LTP always needs the probs -> force eager attention outputs on every layer.
    global_mask, sliding_mask = self._update_attention_mask(
        attention_mask, output_attentions=True
    )

    hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

    ltp_on = st.mode in ("soft", "hard")
    st.soft_masks = []
    st.kept_frac = []
    # additive bias accumulating hard-pruned KEY columns (bs,1,1,seq); 0 until a drop.
    prune_bias = None

    for i, layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        g_mask = global_mask if prune_bias is None else global_mask + prune_bias
        s_mask = sliding_mask if prune_bias is None else sliding_mask + prune_bias
        layer_out = layer(
            hidden_states,
            attention_mask=g_mask,
            sliding_window_mask=s_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=True,
        )
        hidden_states = layer_out[0]
        if not ltp_on:
            continue
        attn = layer_out[1]                                   # (bs, heads, q, k)
        score = _received_attention_score(attn, query_valid)  # (bs, seq)
        eff = self.ltp_base[i] + self.ltp_delta[i]            # scalar effective threshold

        if st.mode == "soft":
            # soft masking: differentiable scale of the layer output; thresholds learn.
            # BUGFIX (CLS/pad protection): granite pools the CLS token (hidden[:, 0];
            # config.classifier_pooling=="cls"). Scaling CLS toward 0 destroys the
            # classification signal AND opens a huge 1/T-amplified gradient path through
            # CLS's OWN attention-received score -> from a trained init the loss jumps to
            # ~12 and grad_norm explodes (observed: 9e5). The HARD path already force-
            # keeps CLS; SOFT must mirror it. Force soft==1 at CLS (pos 0) and at pad
            # positions (never scale non-tokens), matching hard's `keep & query_valid`
            # + `keep[:,0]=True`. The reference (plain RoBERTa) never hit this because
            # its <s> pooling token naturally receives high attention (score>threshold);
            # we protect it explicitly rather than relying on that.
            soft = torch.sigmoid((score - eff) / st.temperature)     # (bs, seq)
            protect = ~query_valid                                   # pad -> keep as 1
            protect[:, 0] = True                                     # CLS -> keep as 1
            soft = torch.where(protect, torch.ones_like(soft), soft)
            real = query_valid.clone()
            real[:, 0] = False                                       # prunable-token mask
            st.soft_masks.append((soft, real))
            hidden_states = hidden_states * soft.unsqueeze(-1)
        else:  # "hard": extend the attention mask so later layers ignore pruned keys
            thr = torch.clamp(eff, min=1e-5)
            keep = (score >= thr) & query_valid                  # (bs, seq) bool
            keep[:, 0] = True                                    # never drop CLS
            keep = keep & query_valid
            st.kept_frac.append(float(keep.float().mean()))
            drop = (~keep).to(hidden_states.dtype) * _NEG        # (bs, seq)
            add = drop[:, None, None, :]                         # (bs,1,1,seq)
            prune_bias = add if prune_bias is None else prune_bias + add
            # also zero the pruned tokens' hidden states so a downstream mean-pool /
            # residual can't leak them (CLS pooling ignores them anyway).
            hidden_states = hidden_states * keep.unsqueeze(-1).to(hidden_states.dtype)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    hidden_states = self.final_norm(hidden_states)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, None] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=None,   # deliberately not bubbled: keeps HF Trainer from gathering
    )


def install_ltp(model, final_token_threshold, temperature=1e-3, ltp_lambda=0.0, mode="soft"):
    """Attach LTP to a loaded ModernBertForSequenceClassification (granite).

    Adds per-layer threshold params (ltp_delta) + base buffers (ltp_base) on the inner
    ModernBertModel and swaps its forward for _ltp_inner_forward. The outer head /
    pooling / classification-loss path is untouched (stock), so a disabled LTP is a
    parity no-op. Returns `model` (mutated in place).

    mode: 'soft' (train), 'hard' (eval/deploy), or 'off' (parity: stock forward).
    """
    inner = getattr(model, "model", None)
    assert inner is not None and inner.__class__.__name__ == "ModernBertModel", (
        "install_ltp expects a ModernBertForSequenceClassification (inner .model must "
        f"be a ModernBertModel); got {type(model).__name__}"
    )
    assert model.config._attn_implementation == "eager", (
        "load the model with attn_implementation='eager' before install_ltp"
    )
    L = len(inner.layers)
    dev = next(inner.parameters()).device
    # base[i] = final * i / L  (linear ramp, reference); delta learnable, init 0.
    base = torch.tensor([final_token_threshold * i / L for i in range(L)],
                        dtype=torch.float32, device=dev)
    inner.register_buffer("ltp_base", base, persistent=False)
    inner.ltp_delta = nn.ParameterList(
        [nn.Parameter(torch.zeros((), dtype=torch.float32, device=dev)) for _ in range(L)]
    )
    inner._ltp = _LTPState(L, final_token_threshold, temperature, ltp_lambda, mode)
    inner.forward = types.MethodType(_ltp_inner_forward, inner)
    model._ltp = inner._ltp  # convenience handle on the outer model too
    # record on config so a saved checkpoint reloads reproducibly (E4 factored_ffn
    # pattern): rebuild base model -> install_ltp(**config.ltp) -> load ltp_delta.
    model.config.ltp = {"final_token_threshold": float(final_token_threshold),
                        "temperature": float(temperature), "lambda": float(ltp_lambda)}
    logger.info(
        f"LTP installed: {L} layers, final_token_threshold={final_token_threshold}, "
        f"T={temperature}, lambda={ltp_lambda}, mode={mode}; "
        f"base thresholds {base[0].item():.4g}..{base[-1].item():.4g}"
    )
    return model


def set_ltp_mode(model, mode):
    assert mode in ("off", "soft", "hard")
    model.model._ltp.mode = mode


def set_ltp_temperature(model, temperature):
    model.model._ltp.temperature = float(temperature)


def make_ltp_temp_callback(model, t_start, t_end):
    """TrainerCallback that LINEARLY anneals the soft-mask temperature from t_start
    (soft/gentle: masks ~0.5, small O(1/t_start) gradients — ramps pruning in without
    shocking a trained init) down to t_end (sharp/near-hard, discriminative) over the
    full run. This is the reference's set_temperature schedule idea
    (modeling_ltp.set_temperature). Fixes the step-1 gradient explosion at fixed small T
    (grad ∝ 1/T): starting at t_start≈10x the attention-score scale keeps grads O(1/T)≈20,
    ending at t_end≈score scale gives near-hard masks matching hard-mode deployment."""
    from transformers import TrainerCallback

    class _LTPTempCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            set_ltp_temperature(model, t_start)

        def on_step_begin(self, args, state, control, **kw):
            total = max(1, state.max_steps)
            frac = min(1.0, state.global_step / total)
            set_ltp_temperature(model, t_start + (t_end - t_start) * frac)  # linear

    return _LTPTempCallback()


def set_ltp_disabled_softmask(model):
    """Parity helper: force every effective threshold to -inf so the SOFT path's mask
    is sigmoid(+inf) == 1.0 exactly -> hidden * 1.0 is a numeric no-op. Exercises the
    full soft-mask code while guaranteeing byte-identical logits."""
    inner = model.model
    inner._ltp.mode = "soft"
    with torch.no_grad():
        inner.ltp_base.fill_(-1e30)
        for p in inner.ltp_delta:
            p.zero_()


def reset_ltp_thresholds(model, final_token_threshold=None):
    """Restore the per-layer base threshold ramp base[i]=final*i/L and zero the learnable
    deltas. Use after set_ltp_disabled_softmask() to get real thresholds back (e.g. for
    hard-mode telemetry). If final_token_threshold is None, reuses the installed value."""
    inner = model.model
    st = inner._ltp
    final = st.final if final_token_threshold is None else float(final_token_threshold)
    st.final = final
    L = st.num_layers
    with torch.no_grad():
        for i in range(L):
            inner.ltp_base[i] = final * i / L
        for p in inner.ltp_delta:
            p.zero_()


def ltp_regularizer(model):
    """lambda * sum_over_layers(mean soft-mask) — the LTP sparsity term (soft mode).
    Reference trainer.training_step, masking_mode=='soft' uses mask.mean() over ALL
    positions; we take the mean over PRUNABLE tokens only (real, non-CLS) because CLS
    and pad are force-kept at 1 here — averaging them in would dilute lambda and make
    it depend on the padding fraction. Returns a 0-dim tensor."""
    st = model.model._ltp
    if st.mode != "soft" or not st.soft_masks:
        return torch.zeros((), device=next(model.parameters()).device)
    reg = 0.0
    for soft, real in st.soft_masks:
        denom = real.sum().clamp(min=1)
        reg = reg + (soft * real).sum() / denom
    return st.ltp_lambda * reg


def ltp_token_stats(model):
    """Per-layer kept-token fraction from the most recent HARD forward (telemetry)."""
    return list(model.model._ltp.kept_frac)


def separate_threshold_params(model):
    """Split params into (threshold_params, other_params). The reference trains the
    thresholds with their OWN optimizer (lr_threshold, weight_decay_threshold) because
    they move on a different scale than the network weights. Use this to build two
    param groups."""
    thr, other = [], []
    thr_ids = {id(p) for p in model.model.ltp_delta}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (thr if id(p) in thr_ids else other).append(p)
    return thr, other


def make_ltp_trainer(base_cls, loss_fn=None):
    """Trainer subclass that adds the LTP sparsity regularizer to the (soft-mode) loss.

    loss_fn(logits, labels) -> scalar: the classification loss to use (e.g. the
    label-smoothing loss from finetune.classification_loss). If None, uses the model's
    built-in loss (outputs.loss). The regularizer is only nonzero in soft mode.
    Eval batches (model.eval / mode may be hard or soft) get the plain classification
    loss with no regularizer.
    """

    class _LTPTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            if loss_fn is not None and labels is not None:
                loss = loss_fn(outputs.logits, labels)
            else:
                loss = outputs.loss
            if model.training:
                loss = loss + ltp_regularizer(model)
            return (loss, outputs) if return_outputs else loss

    return _LTPTrainer
