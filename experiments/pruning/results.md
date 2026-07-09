# Branch: pruning — compress the champion without losing accuracy
Branch: `kyusang_kvprune_svd` · logical group `research/pruning` (the per-branch topology was planned but never created — all work is committed on kyusang_kvprune_svd) · Baseline: **E8b qwen3 richargs full-FT = 0.7643 uncal** (E3 probe baseline = champion rank-1.0 = 0.7734). All scores uncalibrated macro-F1.

## Summary
Legend: ✅ win · ❌ no-go/closed · 🟢 ready, not run.

| Exp | Experiment | Model | Status | Result | Δ vs baseline | Verdict |
|---|---|---|:--:|---|---|---|
| E3 | FFN whitened-SVD probe (training-free) | qwen3 | ✅ | −0.26pt @ r=512, +0.42pt @ r=768; cliff < r≈384 | vs rank-1.0 0.7734 | GO — greenlit E4 |
| E4 | FFN SVD prune r=512 + recovery-FT | qwen3 | ✅ | **0.7688** | **+0.0045** | WIN — net-positive, ~15% params cut |
| E16 | qwen3 depth-prune 28→14 + recovery | qwen3 | ✅ | **0.7638** | **−0.0005** | WIN — ~free, half depth → ~2× faster |
| E16b | granite depth-prune the best model | granite | ⚠️ VOID | 0.1602 / 0.2365 (1-LAYER model) | — | **INVALID — implementation bug, not a finding** (see below) |
| E18 | LTP token pruning | qwen3 (planned) | 🟢 | — | — | READY, not yet run |

## E3 — FFN whitened-SVD probe (training-free)
- **Model:** qwen3-0.6B (vocab-pruned champion).
- **What:** measure how low-rank the FFN can go with activation-aware (whitened) SVD, no retraining, to decide whether E4 is worth running.
- **Baseline:** unmodified model, rank-1.0 = 0.7734.
- **Change:** truncate every FFN gate/up/down matrix to rank *r* via whitened SVD; no fine-tune.
- **Result:** r=768 **+0.0042** (mild denoise), r=512 **−0.0026**, r=384 −0.0135, r=256 −0.068 (cliff below r≈384). FFN = 44% of model params.
- **Verdict:** ✅ **GO** — FFN factors cheaply and is the biggest param block; greenlit E4.

## E4 — FFN SVD prune (r=512) + recovery-FT
- **Model:** qwen3-0.6B (E8b champion).
- **What:** ship a compressed qwen3 by factoring the FFN and recovering with a short fine-tune.
- **Baseline:** E8b qwen3 = 0.7643 (uncompressed).
- **Change:** FFN → whitened-SVD rank 512 (~33% of FFN, ~15% of whole model cut) + 2ep recovery @ lr 5e-6.
- **Result:** **0.7688** (~85% params) = **+0.0045**.
- **Verdict:** ✅ **WIN** — compression is net-POSITIVE (smaller *and* better). Checkpoint: `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e4_ffn_r512_recover_e8b/checkpoint-4157`. Figure: `figures/e4_recovery.png`.

## E16 — qwen3 depth-prune 28→14 + recovery-FT
- **Model:** qwen3-0.6B (E8b champion).
- **What:** drop half the transformer layers for ~2× faster inference, then recover.
- **Baseline:** E8b qwen3 = 0.7643 (28 layers).
- **Change:** keep the 14 highest block-influence layers `[0-7,9-11,19,21,27]` (ShortGPT selection), drop the 14 most redundant + recovery-FT (E4 recipe).
- **Result:** **0.7638** at HALF depth = **−0.0005**.
- **Verdict:** ✅ **WIN** — depth-prune is ~free on qwen3. Checkpoint: `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e16_qwen3_depth14_recover/checkpoint-8312` (loads with standard `from_pretrained`; SUBMISSIONS.md → **E16z**).

## E16b — depth-prune the BEST model (granite E8a+LS 0.7803) — ⚠️ VOID (BUG)
> **AUDIT 2026-07-08: this experiment is INVALID — an implementation bug, not a scientific NO-GO.**
> The two runs passed **`--keep_layer_idx 11`** and **`--keep_layer_idx 14`**. In `src/finetune.py:777-778`
> that flag is an explicit *index list*, so `prune_layers(model, [11])` / `[14]` kept **exactly ONE layer**,
> not 11/14 layers. Confirmed: both checkpoints have `config.num_hidden_layers=1` (vs qwen3 E16's 14);
> the launch log reads `pruned encoder depth 22 -> 1 (kept layers [11])`. The correct flag is `--keep_layers 11`
> (a count). So 0.1602/0.2365 is the score of a **1-layer granite** (≈ random for 14 classes) — the intended
> 11/14-layer granite was **never trained**. qwen3 E16 worked because it correctly passed the full 14-index list.
> **The recorded "granite layers are load-bearing / backbone-dependent / granite can't depth-prune" conclusion is
> false-attributed to a model that never existed and must be VOIDED.** To actually test granite depth-prune, rerun
> with `--keep_layers`. (Secondary latent bug even for a correct rerun: `prune_layers` slices the ModuleList without
> repairing ModernBERT's global-attention-every-3 pattern or per-layer RoPE-θ → the pruned config's
> `global_attn_every_n_layers` would no longer align with the kept indices. Fix before trusting any granite depth-prune.)
> Code fix is HELD pending user go-ahead.
- **Model:** granite-311m (E8a+LS champion).
- **What (intended):** apply the E16 depth-prune trick to the champion (granite) to make it fast+small.
- **Baseline:** granite E8a+LS = 0.7803 (22 encoder layers).
- **What actually ran (BUGGED):** `--keep_layer_idx 11` / `14` → kept a SINGLE layer (`num_hidden_layers=1`); intended keep-11/keep-14 never ran.
- **Result:** **0.1602** / **0.2365** = a **1-layer** granite (expected ≈ random). NOT evidence about 11/14-layer granite.
- **Verdict:** ⚠️ **VOID — no conclusion about granite depth-prune tolerance is supported.** Requires a corrected `--keep_layers` rerun. (Shipping is unaffected — the champion is already fast at 5:10.)

## E18 — LTP token pruning
- **Model:** qwen3-0.6B (planned target).
- **What:** drop unimportant tokens mid-network for ~2× throughput (<1% drop on RoBERTa-class).
- **Baseline:** same model without token pruning.
- **Change:** learned per-layer token-drop threshold.
- **Result:** — **not run**.
- **Verdict:** 🟢 READY (user-approved), not yet started.

## Notes
- Param split (qwen3-0.6B): FFN 44.4%, embeddings 26.1%, attention 29.6%. FFN is the prize; embeddings → vocab-prune (separate proven recipe); K/V-SVD is only ~10% of params.
- "Whitened SVD" = activation-aware (SVD-LLM/Palu) truncation; ≫ plain SVD; training-free always degrades, prune→recover can be net-zero or better.
- E3 gotcha: the pruned config ships `pad_token_id=25284` (pruned-space); do NOT override it or last-token pooling breaks (−0.10 F1).
- E16 layer selection matches the official ShortGPT block-influence (`github.com/icip-cas/ShortGPT`) at the **formula level only** — ⚠️ the official repo was NOT cloned and diffed (paper-method invariant not fully met); one intended divergence — we mask BI to non-pad tokens. (E4's whitened-SVD, by contrast, WAS diffed vs the cloned `AIoT-MLSys-Lab/SVD-LLM@7538cca`.)
- ⚠️ Figure invariant: **E16 and E16b have no figure** in `figures/` (only E3 `ffn_svd_degradation.png` and E4 `e4_recovery.png` exist). E16 (a WIN) should get a depth-prune recovery figure.
