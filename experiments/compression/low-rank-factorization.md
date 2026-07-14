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

## qwen3_ls (E45) — SVD-degradation probe DONE 2026-07-14 (training-free)

Probe on the **qwen3_ls baseline** (`analysis/palu_probe_ffn.py`, in-sample subset base
0.8011, deltas robust). ⚠️ **The E4 +0.0045 gain does NOT reproduce training-free on the LS
baseline** — FFN low-rank is a net *cost* here, steeper than E3's no-LS E8b:

| rank | FFN param kept | FFN Δ (this, LS) | FFN Δ (E3, E8b no-LS) | KV Δ |
|---|---|---|---|---|
| 768 | 100% | −0.0030 | +0.0042 (denoised) | +0.0007 |
| 640 | 83% | −0.0049 | — | +0.00001 |
| 512 | **67%** | **−0.0106** | −0.0026 | **+0.0012** |
| 384 | 50% | −0.0346 | −0.0135 | −0.0010 |
| 256 | 33% | −0.101 | −0.0676 | −0.0147 |

- **FFN:** no free denoising on qwen3_ls; r=512 costs −0.0106 for a 33% FFN-param cut.
  E4's net-positive came from *recovery* turning −0.0026 → +0.0045 (a +0.007 swing); the LS
  hole starts deeper (−0.0106), so reproducing net-positive is uncertain — the recovery arm
  decides. Milder r=640/768 (−0.005/−0.003) are the safer recovery candidates.
- **RECOVERY ARM DONE 2026-07-14 (r=512 + 2ep LS, from qwen3_ls, trainer eval on 3.5k
  held-out): 0.76097 = net −0.0047** vs 0.76569. Recovery closed the −0.0106 zero-shot hole
  by ~0.006 but stayed NEGATIVE — **the E4 +0.0045 gain does NOT reproduce on the LS
  baseline** (r640/r768 arms killed early per user: won't submit qwen3). **qwen3 line CLOSED:
  neither depth (best keep-14 net −0.002..−0.004) nor FFN low-rank (−0.0047) improves
  qwen3_ls; the E4 gain was specific to the no-LS E8b.**
- **KV:** mildly denoises (+0.0012 @ r512) but only saves params at r≤384, and attention is
  ~10% of qwen3 — minor tile (E45 B3 is more about q/o than KV).
- ⚠️ **Blocker for the recovery arm:** `--factor_ffn` calibration (`build_factored_model`)
  assumes a **full-vocab** checkpoint — it tokenizes calib texts in full space, but qwen3_ls
  is vocab-pruned, so the calibration forward needs remap-awareness (or a full-vocab qwen3_ls
  ckpt). Must wire remap into the factor calibration before the recovery arm can run.

## granite — expected low yield

granite's FFN is only 18.7% of params and already thin (I=1152=1.5·H), and attention is
four small H² projections. So low-rank has little mass to win and the speed payoff is
marginal. Run the E3-style whitened-SVD degradation probe on granite **only if** the
depth ([depth-pruning.md](depth-pruning.md)) + width verdicts leave a gap to close.

## Combination
qwen3: E4 FFN-factor is claimed to STACK with E16 depth-prune but never combined — that
product is E45's core bet ([results.md](results.md) qwen3 track).
