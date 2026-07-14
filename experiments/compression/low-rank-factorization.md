# Low-rank factorization (structured) — whitened-SVD factor + recovery FT

**Method:** replace a weight matrix `W` with a factored `down·up` pair at rank r, init =
activation-aware (whitened) SVD, then recovery FT. SVD-LLM style. Code: `src/factored_ffn.py`,
`--factor_ffn`; whitening validated against `AIoT-MLSys-Lab/SVD-LLM@7538cca` (diff = MATCH).
Index: [results.md](results.md).

| Model | Result | Value | Status |
|---|---|---|---|
| **qwen3** FFN r=512 | 0.7688 vs 0.7643 unpruned = **+0.0045 (net-POSITIVE)** | ~15% params; size/regularization win | ✅ done (E3/E4) |
| **granite** FFN | not tried; expected LOW yield | FFN thin (768↔1152, 18.7%) | 🔲 probe only if stack-compression needed |

## qwen3 — E3/E4 (done)

- **E3 probe (training-free, whitened-SVD degradation):** FFN low-rank: −0.26pt @ r=512
  (66% params kept), **+0.42pt @ r=768** (whitening *denoises* above baseline), cliff below
  r≈384 (−1.35pt @ 50% params). FFN = 44% of qwen3 params = the compression prize.
- **E4 (factor r=512 + 2ep recovery FT):** 0.7688 uncal, **+0.0045 over uncompressed E8b
  0.7643**, ~15% fewer whole-model params. Low-rank FFN acts as a regularizer.
  `from_pretrained` round-trip bit-exact.
- ⚠️ ms/sample was never measured — "little acceleration" is correct: this was a **size +
  accuracy** win, not a speed one. The GEMM is smaller but the win was regularization.

## granite — expected low yield

granite's FFN is only 18.7% of params and already thin (I=1152=1.5·H), and attention is
four small H² projections. So low-rank has little mass to win and the speed payoff is
marginal. Run the E3-style whitened-SVD degradation probe on granite **only if** the
depth ([depth-pruning.md](depth-pruning.md)) + width verdicts leave a gap to close.

## Combination
qwen3: E4 FFN-factor is claimed to STACK with E16 depth-prune but never combined — that
product is E45's core bet ([results.md](results.md) qwen3 track).
