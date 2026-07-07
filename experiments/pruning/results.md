# Branch: pruning — compress the champion without losing accuracy
Git branch: `research/pruning` · Experiments: E3 (FFN whitened-SVD probe), E4 (SVD prune + recovery FT, +E4b Minitron distill-recovery), E16 (depth-prune w/ ShortGPT selection, LaCo fallback), E18 (LTP token pruning, ToMe fallback)
Method decision tree (user 2026-07-07): ShortGPT now · LaCo if ShortGPT fails ·
SliceGPT if E3+E4 fail · Minitron distill-recovery after E4 succeeds (baseline = E4's
own result) · LTP now, ToMe if LTP fails · Sheared-LLaMA skipped (data scale) ·
2:4 sparsity skipped (unknown eval GPU, no zip benefit, runtime risk) · E3 = 3-scorer probe (whitened-SVD vs Wanda-sp vs FLAP) · CFSP dropped (complexity) · EoRA = recovery tier in E4 ladder + THE recovery for E15d translator
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

### 2026-07-08 · E4 — FFN whitened-SVD prune (r=512) + recovery-FT · **WIN**
Took the E8b champion (qwen3 richargs full-FT, unfactored) and factored every FFN
gate/up/down to **rank 512** (whitened / activation-aware SVD — the E3-validated
truncation), then ran a short recovery fine-tune. FFN is 44% of params; r=512 removes
~33% of FFN params ≈ **~15% of the whole model**. Recovery FT: `full_ft`, linear head,
**2 epochs @ lr 5e-6**, max_len 512, serialize richargs, `init_from` the E8b checkpoint
(`checkpoint-8314`), full_data. Uncalibrated `val_macro_f1` (calibrated column ignored).

| model | params | val_macro_f1 | Δ vs E8b |
|---|---|---|---|
| E8b — unfactored (full FFN) | 100% | **0.7643** | — |
| **E4 — FFN r=512 + recovery** | **~85%** | **0.7688** | **+0.0045** |

**Read-out → E4 is a WIN. Compression is net-POSITIVE.** The compressed model *beats*
the uncompressed champion (+0.0045) while shedding ~15% of params: recovery-FT more than
closes the −0.26pt training-free truncation gap (E3), and the low-rank FFN acts as a mild
denoiser on top. This is a ship-able compressed qwen3.
> Not directly comparable: the E3 training-free immediate at r=512 was 0.7707, but on a
> different subset/model (the vocab-pruned champion). The clean, same-recipe comparison is
> **E4 0.7688 vs E8b 0.7643**.

Figure: [figures/e4_recovery.png](figures/e4_recovery.png)
Sources: `output/pat/ft_results_e4.csv` (tag `e4_ffn_r512_recover_e8b`), `output/pat/ft_results_e8b.csv` (tag `e8b_qwen3_richargs_full`)

### 2026-07-08 · E16 — qwen3 depth-prune 28→14 (ShortGPT Block-Influence) + recovery-FT · **LAUNCHED (pending harvest)**
Depth-halve the E8b champion (28→14 transformer layers) via **ShortGPT** layer selection,
then recovery-FT with the E4 recipe.

**Paper-method invariant (ShortGPT diff).** Official repo = **github.com/icip-cas/ShortGPT**
(`shortgpt/metrics.py::block_influence`, `shortgpt/shortgpt.py`). Their BI:
```python
sim = (h_in @ h_out.T)/(||h_in||·||h_out||); sim = sim.diagonal(); return 1 - sim
```
accumulated per layer (`importances[i] += block_influence(hiddens[i], hiddens[i+1]).sum()`),
then `remove_layers` drops `argsort(importances)[:n_prune]` (the **lowest-BI = most
redundant / identity-like** layers). **Our implementation MATCHES** the canonical single-layer
(n=1) BI: `BI_i = mean_tokens(1 − cos(hidden_states[i], hidden_states[i+1]))`, drop the 14
lowest. **One faithful divergence:** we mask BI to non-pad tokens via `attention_mask` (official
reshapes all positions incl. pad — but it runs unpadded causal-LM stride windows; here padding
would pollute cosine). Sum-vs-mean is rank-invariant. Selection script:
`experiments/pruning/e16_shortgpt_bi.py` (300 richargs val-eval samples, 76,565 tokens, GPU 3).

**BI profile (mean 1−cos over calib tokens):** layer 0 = 0.925 and last layer 27 = 0.747
dominate (do the most work, least redundant); a contiguous mid/late block (12–26) is the most
redundant — the textbook ShortGPT shape.
- **KEEP (14, highest BI):** `[0,1,2,3,4,5,6,7,9,10,11,19,21,27]`
- **DROP (14, lowest BI):** `[8,12,13,14,15,16,17,18,20,22,23,24,25,26]`

The kept set is **non-contiguous** (drops L8, keeps L9–11; keeps L19/L21/L27 amid a dropped
tail), so evenly-spaced/first-14 pruning cannot express it. Added `--keep_layer_idx` to
`src/finetune.py` (`prune_layers` now accepts an explicit index list; the old int
evenly-spaced path is unchanged) to build the exact BI set from `--init_from` the E8b
checkpoint.

**Recovery-FT launch (E4 recipe):** full_ft, linear head, **2ep @ lr 5e-6**, richargs,
full_data, `--loss ce`, max_len 512, init_from `checkpoint-8314`, GPU 3.
Tag `e16_qwen3_depth14_recover`, results → `output/pat/ft_results_e16.csv`.
Launcher: `experiments/pruning/e16_launch.sh`; log: `sbatch/logs/e16-depth14-recover.out`.

**Harvest plan:** `ft_results_e16.csv` uncal `val_macro_f1` vs **E8b 0.7643**. 14-layer =
half the depth → ~2× faster inference. Read-out: ≈0.7643 → depth-prune WIN (big speed/budget
gain at ~same acc); if it craters (≫1pt below) → LaCo layer-merge fallback or fewer dropped
layers. (Run in progress at time of writing; NFS-slow warmup.)
