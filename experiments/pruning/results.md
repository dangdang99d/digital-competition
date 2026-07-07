# Branch: pruning — compress the champion without losing accuracy
Git branch: `research/pruning` · Experiments: E3 (FFN whitened-SVD probe), E4 (SVD prune + recovery FT)
Baseline: Qwen3-0.6B full-FT v1@512 = 0.7682 uncal (E3: same-subset rank-1.0 row).

## Prior findings (from main-session analysis, 2026-07-07)
- KV whitened-SVD probe (training-free, 3k val): 95.5% energy @ 25% rank; −0.005 @ r=512, −0.010 @ r=256. Method works where naive width50 slice failed.
- BUT K+V = only 9.9% of params. Split: FFN 44.4%, embeddings 26.1%, attn 29.6% (Q/O 19.7%). FFN is the prize; embeddings → vocab-prune (proven recipe).
- Palu/ASVD/SVD-LLM lit: activation-aware (whitened) SVD ≫ plain; training-free always degrades; prune→recover regime (ours) can be net-zero or better.

## Results
(append: date · experiment · numbers vs baseline · [figure](figures/...) · commit)

### 2026-07-07 · E3 — FFN whitened-SVD probe (training-free) · **E4 = GO**
Ran `analysis/palu_probe_ffn.py` (extends `palu_probe.py` to gate/up/down + reuses K/V) on the
**vocab-pruned champion** from `submissions/submit_0703_qwen3_pruned.zip` (unzipped to local disk;
FFN/attn/head byte-identical + parity-preserving). GPU 1, 3k-val subset, uncal argmax macro-F1,
n_calib=256, max_len 512, variant v1. Whitened (activation-aware, SVD-LLM) truncation,
rank ratio = r / min(out,in). Figure: [figures/ffn_svd_degradation.png](figures/ffn_svd_degradation.png).

**Parity gate PASSED.** rank-1.0 (unmodified) = **0.7734** (== reference 0.7734). K/V control curve
reproduces the prior validated finding exactly: r=512 **−0.0051**, r=256 **−0.0103** @ 95.5% energy
(prior: −0.005 / −0.010 @ 95.5%). Forward pass is correct.
> Bug found & fixed en route: the probe must NOT override `model.config.pad_token_id`. The pruned
> config ships `pad_token_id=25284` (a PRUNED-space row = `remap[eos_full=151643]`); after `remap` the
> padded positions hold 25284, so Qwen3 last-token pooling works. Overriding it to full-space eos
> (151643, as `palu_probe.py` does — valid there since it never remaps) silently broke pooling →
> −0.10 macro-F1 (an early run read a bogus 0.6745). Fix mirrors the verified `script.py` (no override).

**FFN whitened-SVD curve (gate/up/down together), Δ vs 0.7734 baseline:**

| ratio | rank | energy | params kept/matrix | macro-F1 | Δ |
|------:|-----:|-------:|-------------------:|---------:|------:|
| 1.000 | 1024 | 100.0% | 100.0% | 0.7734 | +0.0000 |
| 0.875 |  896 |  99.8% | 100.0% | 0.7758 | **+0.0024** |
| 0.750 |  768 |  99.2% | 100.0% | 0.7776 | **+0.0042** |
| 0.625 |  640 |  98.3% |  83.3% | 0.7749 | +0.0015 |
| 0.500 |  512 |  96.8% |  66.7% | 0.7707 | **−0.0026** |
| 0.375 |  384 |  94.5% |  50.0% | 0.7598 | −0.0135 |
| 0.250 |  256 |  90.7% |  33.3% | 0.7058 | −0.0676 |

**Read-out → E4 is GO with FFN as the compression target.** FFN is even MORE low-rank than K/V: at
50% rank (r=512) it is **−0.26pt** (training-free, no recovery), far inside the "−1pt @ ≤50% rank"
GO bar; it stays flat/above baseline (whitened truncation acts as a mild denoiser — +0.4pt at r=768)
until it cliffs below r≈384. At 50% *params* kept (r=384) it is only −1.35pt training-free, which E4's
recovery FT should close to net-zero. FFN is 44% of params vs K/V's ~10%, so this is the real prize:
factorizing FFN to r=512 saves ~33% of FFN params (~15% of the whole model) at −0.26pt before any
recovery. Suggested E4 start: FFN rank 384–512 + short recovery FT.

Data: `scratchpad/e3_ffn_results.json` (session-local). Not committed (working tree mid-reorg).
