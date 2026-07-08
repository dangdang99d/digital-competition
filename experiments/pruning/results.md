# Branch: pruning — compress the champion without losing accuracy
Git branch: `research/pruning` · Baseline: **E8b qwen3 richargs full-FT = 0.7643 uncal** (E3 probe baseline = champion rank-1.0 = 0.7734). All scores uncalibrated macro-F1.

## Summary
Legend: ✅ win · ❌ no-go/closed · 🟢 ready, not run.

| Exp | Experiment | Status | Result | Δ vs baseline | Verdict |
|---|---|:--:|---|---|---|
| E3 | FFN whitened-SVD probe (training-free) | ✅ | −0.26pt @ r=512, +0.42pt @ r=768; cliff < r≈384 | vs rank-1.0 0.7734 | GO — greenlit E4 |
| E4 | FFN SVD prune r=512 + recovery-FT | ✅ | **0.7688** | **+0.0045** | WIN — net-positive, ~15% params cut |
| E16 | qwen3 depth-prune 28→14 + recovery | ✅ | **0.7638** | **−0.0005** | WIN — ~free, half depth → ~2× faster |
| E16b | granite depth-prune the best model | ❌ | 0.1602 / 0.2365 | ≪ 0.7803 | NO-GO — granite won't recover |
| E18 | LTP token pruning | 🟢 | — | — | READY, not yet run |

## E3 — FFN whitened-SVD probe (training-free)
- **What:** measure how low-rank the FFN can go with activation-aware (whitened) SVD, no retraining, to decide whether E4 is worth running.
- **Baseline:** unmodified model, rank-1.0 = 0.7734.
- **Change:** truncate every FFN gate/up/down matrix to rank *r* via whitened SVD; no fine-tune.
- **Result:** r=768 **+0.0042** (mild denoise), r=512 **−0.0026**, r=384 −0.0135, r=256 −0.068 (cliff below r≈384). FFN = 44% of model params.
- **Verdict:** ✅ **GO** — FFN factors cheaply and is the biggest param block; greenlit E4.

## E4 — FFN SVD prune (r=512) + recovery-FT
- **What:** ship a compressed qwen3 by factoring the FFN and recovering with a short fine-tune.
- **Baseline:** E8b qwen3 = 0.7643 (uncompressed).
- **Change:** FFN → whitened-SVD rank 512 (~33% of FFN, ~15% of whole model cut) + 2ep recovery @ lr 5e-6.
- **Result:** **0.7688** (~85% params) = **+0.0045**.
- **Verdict:** ✅ **WIN** — compression is net-POSITIVE (smaller *and* better). Checkpoint: `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e4_ffn_r512_recover_e8b/checkpoint-4157`. Figure: `figures/e4_recovery.png`.

## E16 — qwen3 depth-prune 28→14 + recovery-FT
- **What:** drop half the transformer layers for ~2× faster inference, then recover.
- **Baseline:** E8b qwen3 = 0.7643 (28 layers).
- **Change:** keep the 14 highest block-influence layers `[0-7,9-11,19,21,27]` (ShortGPT selection), drop the 14 most redundant + recovery-FT (E4 recipe).
- **Result:** **0.7638** at HALF depth = **−0.0005**.
- **Verdict:** ✅ **WIN** — depth-prune is ~free on qwen3. Checkpoint: `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e16_qwen3_depth14_recover/checkpoint-8312` (loads with standard `from_pretrained`; SUBMISSIONS.md → **E16z**).

## E16b — depth-prune the BEST model (granite E8a+LS 0.7803)
- **What:** apply the E16 depth-prune trick to the champion (granite) to make it fast+small.
- **Baseline:** granite E8a+LS = 0.7803 (22 encoder layers).
- **Change:** keep 11 then 14 layers (evenly-spaced) + recovery-FT (lr 5e-6, then lr 2e-5/3ep).
- **Result:** **0.1602** (keep-11) / **0.2365** (keep-14) — both near-random (14-class floor ≈ 0.07).
- **Verdict:** ❌ **NO-GO** — granite/ModernBERT layers are load-bearing; recovery can't rebuild a halved encoder. Depth-prune tolerance is backbone-dependent (qwen3 free, granite fails). Moot for shipping — the champion is already fast (5:10).

## E18 — LTP token pruning
- **What:** drop unimportant tokens mid-network for ~2× throughput (<1% drop on RoBERTa-class).
- **Baseline:** same model without token pruning.
- **Change:** learned per-layer token-drop threshold.
- **Result:** — **not run**.
- **Verdict:** 🟢 READY (user-approved), not yet started.

## Notes
- Param split (qwen3-0.6B): FFN 44.4%, embeddings 26.1%, attention 29.6%. FFN is the prize; embeddings → vocab-prune (separate proven recipe); K/V-SVD is only ~10% of params.
- "Whitened SVD" = activation-aware (SVD-LLM/Palu) truncation; ≫ plain SVD; training-free always degrades, prune→recover can be net-zero or better.
- E3 gotcha: the pruned config ships `pad_token_id=25284` (pruned-space); do NOT override it or last-token pooling breaks (−0.10 F1).
- E16 layer selection matches the official ShortGPT block-influence (`github.com/icip-cas/ShortGPT`); one faithful divergence — we mask BI to non-pad tokens.
