"""E37 — alternative sequence poolings for the granite (ModernBERT) classifier.

Self-contained in the E37 folder — imported by run_e37.py (training) and reusable at
inference. Nothing here touches src/; it operates on a model instance passed in.

The stock ModernBertForSequenceClassification pools ONE token: config.classifier_pooling in
{"cls" (first token), "mean" (masked average)}. cls/mean are native (just set the config field).
This module adds the non-native option:

  attn — learned additive attention pool: a_t = softmax_t(w·h_t) over non-pad tokens,
         out = Σ a_t h_t. "Use every token, let the model weight them" — the transformer-native
         version of the LSTM-pool idea (one attention layer, no recurrence).

install_pooling(model, mode) is the entry point. For "attn" it monkeypatches THIS instance's
forward, reproducing transformers 4.51.3 ModernBertForSequenceClassification.forward verbatim
except the pooling step, and records config.pooling for reload. Padded path only (attn_impl in
{eager,sdpa}; flash-attn unpadded path not used in this project — E19).
"""
import types

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput


class AttentionPool(nn.Module):
    """a_t = softmax_t(w·h_t + b) over non-pad tokens; out = Σ a_t h_t."""

    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, hidden_state, attention_mask):
        # hidden_state [B,T,H], attention_mask [B,T] (1=keep, 0=pad)
        s = self.score(hidden_state).squeeze(-1)                       # [B,T]
        s = s.masked_fill(attention_mask == 0, torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=1).unsqueeze(-1)                      # [B,T,1]
        return (a * hidden_state).sum(dim=1)                           # [B,H]


def _attn_forward(
    self,
    input_ids=None, attention_mask=None, sliding_window_mask=None, position_ids=None,
    inputs_embeds=None, labels=None, indices=None, cu_seqlens=None, max_seqlen=None,
    batch_size=None, seq_len=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, **kwargs,
):
    """Verbatim ModernBertForSequenceClassification.forward (transformers 4.51.3) with the
    single-token pooling replaced by self.attn_pool. Loss branch kept so the default Trainer
    path is identical to stock; custom aux/LS trainers recompute from logits anyway."""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    self._maybe_set_compile()

    outputs = self.model(
        input_ids=input_ids, attention_mask=attention_mask,
        sliding_window_mask=sliding_window_mask, position_ids=position_ids,
        inputs_embeds=inputs_embeds, indices=indices, cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen, batch_size=batch_size, seq_len=seq_len,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    last_hidden_state = outputs[0]                                   # [B,T,H] padded
    pooled = self.attn_pool(last_hidden_state, attention_mask)      # <-- only change vs stock

    pooled_output = self.head(pooled)
    pooled_output = self.drop(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"
        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 \
                else loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss = BCEWithLogitsLoss()(logits, labels)

    if not return_dict:
        output = (logits,)
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss, logits=logits,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions,
    )


def install_attention_pooling(model):
    """Attach an AttentionPool and rebind this instance's forward. ModernBERT only."""
    assert getattr(model.config, "model_type", "") == "modernbert", \
        "attn pooling is implemented for granite/ModernBERT only"
    assert hasattr(model, "head") and hasattr(model, "classifier"), \
        "expected ModernBertForSequenceClassification (head + classifier)"
    ref = model.classifier.weight
    model.attn_pool = AttentionPool(model.config.hidden_size).to(
        device=ref.device, dtype=ref.dtype)
    model.attn_pool.apply(model._init_weights)
    model.forward = types.MethodType(_attn_forward, model)
    model.config.pooling = "attn"          # reload marker (mirrors config.custom_head)
    return model


def install_pooling(model, mode):
    """Entry point. mode in {cls, mean, attn}. cls = no-op (stock)."""
    if mode == "cls":
        model.config.classifier_pooling = "cls"        # explicit; already the default
    elif mode == "mean":
        assert getattr(model.config, "model_type", "") == "modernbert", \
            "mean pooling here targets granite/ModernBERT (native classifier_pooling)"
        model.config.classifier_pooling = "mean"
        model.config.pooling = "mean"
    elif mode == "attn":
        install_attention_pooling(model)
    else:
        raise ValueError(f"unknown pooling mode: {mode}")
    return model
