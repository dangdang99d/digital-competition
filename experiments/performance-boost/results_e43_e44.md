# E43 (weight EMA/SWA) + E44 (granite→qwen3 transfer) — `research/performance-boost`

Detail doc for two arms run together on **vast 44742238** (4×3090, label `E43_E44_ema_qwen3xfer`,
2026-07-13). Summary lives on the [performance-boost board](results.md); this is the full record.
All scores **raw uncalibrated macro-F1** on the full_data 3.5k held-out slice (project invariant —
NO calibration; slice under-predicts & mis-ranks vs LB, **LB is the judge**).

---

## E43 — Weight EMA / SWA on the AWP model — ❌ CLOSED NEGATIVE

**Question:** does within-run temporal weight-averaging (EMA/SWA) add on top of AWP?

**Method:** NOT a training callback — the repo's existing post-hoc path ([[reuse]] `analysis/swa_average.py`):
train the AWP run with `--keep_checkpoints 4` (per-epoch disk checkpoints), then average the
AWP-active epoch tail and eval vs the single best-epoch checkpoint. Fed by a fresh granite AWP run:
```
python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 \
  --serialize richargs --full_data --loss ls \
  --awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0 \
  --batch_size 16 --grad_accum 1 --epochs 4 --seed 42 --init_seed 42 \
  --keep_checkpoints 4 --save_dtype fp32 --tag g_awp_swa
```

**Result (raw F1):**

| model | raw macro-F1 | vs single |
|---|---|---|
| single AWP ckpt (ep4) | 0.7816 | — |
| SWA average ep2–4 | 0.7813 | **−0.0003** (flat) |
| SWA average ep3–4 | 0.7761 | −0.0055 |

**Verdict: discard.** No raw gain; the apparent +0.0023 was **calibration-only** (off-policy). Cause:
only ~4 epochs → no converged tail worth averaging, and **AWP is already a flat-minima method** →
SWA is redundant. User confirmed discard (2026-07-13).

**⚠️ Bug found + fixed:** `analysis/swa_average.py` re-serialized the eval set with `serialize()` =
**v1 default**, not the run's `richargs` → it scored the richargs-trained model at a bogus **0.7317**
(vs true 0.7816) until fixed (added `--serialize`, switched to `build_texts`). Any past swa_average.py
run on a non-v1 (richargs/richmeta) model is invalid.

**Side finding — AWP wants ≥4 epochs.** The granite AWP curve climbs monotonically through ep4:

| epoch | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|
| granite AWP eval_macro_f1 | 0.7264 | 0.7746 | 0.7758 | **0.7822** |

So this run's 0.7822 (ep4) vs E34's 0.7804 (ep3, `--group_by_length`, fp16) = +0.0018 is the **extra
AWP-climbing epoch**, not a recipe improvement (E34 was ep3 = last of its budget). Use ≥4 epochs when
AWP is on; best-epoch selection (finetune.py) makes the extra epoch safe.

---

## E44 — granite→qwen3 backbone transfer of the single-model levers — 🏃 RUNNING

**Question:** do the granite single-model levers (**AWP**, **ELR**) carry to **qwen3-0.6B** — the
backbone that *lost* the slice but *won* the LB (E8b+LS: LB 0.77921 > granite 0.77738)?

**Design — qwen3 2×2** (richargs · full_data · ε=0.1 · ep4 · seed/init 42), one cell per GPU:

| cell | tool | selection | batch | command flags (beyond base) |
|---|---|---|---|---|
| LS+AWP | finetune.py | best-epoch | bs8×ga2 | `--awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0 --keep_checkpoints 4` |
| LS+ELR | elr.py | last-epoch | bs8 | `--ce_mode ls --elr_lambda 3 --elr_beta 0.7` |
| LS+AWP+ELR | elr.py | last-epoch | bs8 | `--elr_lambda 3 --awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0` |

(qwen3 LS alone = reused existing E8b+LS, not retrained.)

**Status / finals (2026-07-13):**

| cell | tool | final |
|---|---|---|
| LS+AWP | finetune, best-epoch | **0.7847** @ep4 ✅ |
| LS+AWP+ELR | elr, ep4 | **0.7788** ✅ |
| LS+ELR | elr, ep4 | **0.7763** ✅ |

**qwen3 LS+AWP 0.7847 is the best qwen3 cell — and it beats granite AWP (finetune) 0.7822 by +0.0025.**
Within the elr loop, AWP adds **+0.0025** (LS+AWP+ELR 0.7788 > LS+ELR 0.7763). ⚠️ the LS+AWP (finetune)
vs LS+AWP+ELR (elr) gap (−0.0059) is mostly the elr-loop code-path deficit (~0.011), not ELR hurting.

### Full granite × qwen3 per-epoch comparison

⚠️ tool / epochs / batch differ per cell (granite 2×2 = elr.py ep3 bs16; granite AWP also has a
finetune ep4 run; qwen3 = ep4, AWP cell finetune bs8×ga2, ELR cells elr bs8). Matched-tool
comparisons flagged below.

**elr.py cells (last-epoch):**

| cell (elr.py) | gr ep1 | gr ep2 | gr ep3 (fin) | qw ep1 | qw ep2 | qw ep3 | qw ep4 (fin) |
|---|---|---|---|---|---|---|---|
| LS (λ0) | 0.6944 | 0.7735 | 0.7690 | — | — | — | *(not re-run for qwen3)* |
| ELR+LS | 0.7090 | 0.7690 | 0.7704 | 0.6873 | 0.7568 | 0.7683 | **0.7763** |
| AWP+ELR+LS | 0.7147 | 0.7718 | 0.7806 | 0.6873 | 0.7646 | 0.7680 | **0.7788** |

**finetune.py AWP cell (best-epoch):**

| cell (finetune) | ep1 | ep2 | ep3 | ep4 (fin) |
|---|---|---|---|---|
| granite AWP | 0.7264 | 0.7746 | 0.7758 | 0.7822 |
| qwen3 AWP | 0.7099 | 0.7638 | 0.7752 | **0.7847** |

**Matched-tool comparison:**

| lever | granite | qwen3 | matched-epoch Δ (gr − qw) |
|---|---|---|---|
| ELR+LS (elr) | 0.7704 (ep3) | 0.7683 (ep3) → 0.7763 (ep4) | **+0.0021** @ep3 (tie) |
| AWP (finetune) | 0.7758 (ep3) / 0.7822 (ep4) | 0.7752 (ep3) / **0.7847** (ep4) | +0.0006 @ep3 → **−0.0025 (qwen3 AHEAD)** @ep4 |
| AWP+ELR (elr) | 0.7806 (ep3) | 0.7680 (ep3) → 0.7788 (ep4) | +0.0126 @ep3 → **+0.0018** at qwen3 ep4 |

**Takeaways (all cells done):**
- **AWP transfers to qwen3 — and qwen3 overtakes granite.** LS+AWP finetune: tied at ep3 (0.7752 vs
  0.7758), then qwen3 out-climbs on ep4 → **qwen3 0.7847 vs granite 0.7822, +0.0025.** The E8 backbone
  edge reappears once qwen3 gets the lever, and the slice *under*-ranks qwen3 → LB gap likely larger.
- **qwen3 needs the 4th epoch:** big ep3→ep4 leaps (AWP +0.0095, AWP+ELR +0.0108, ELR +0.0080) while
  granite plateaus by ep3. Slower-to-warm, higher-ceiling — likely still under its ceiling at ep4 (ep5+
  untested).
- **Lever ranking holds on qwen3:** AWP ≫ ELR; AWP the strongest single cell (0.7847). ELR-on-AWP
  within qwen3 is confounded (finetune vs elr tool) → not cleanly measurable here.
- **LB is the judge** — these are full_data 3.5k-slice numbers. A qwen3 LS+AWP submission is the real
  test; catch = qwen3 ~9:18 inference (needs compression to ship, E40/E45).
</content>
