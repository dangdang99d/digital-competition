# Branch: first-step — the zero-history failure mode
Git branch: `research/first-step` · Experiments: E1 (ceiling A/B: --zero_history vs --strip_history)
Baseline: hist0 generalist on the 1,807 first-step val slice = 0.555 macro-F1 (qwen3: 0.573).

## Prior findings
- step==1 ⟺ len(history)==0 exactly (9,000/70,000; 7,193 train / 1,807 val).
- First-step err ~2× mid-session (36.7% vs ~18%). Both models prior-collapse onto
  list_directory/plan_task; read/grep/glob→list_directory is the dominant confusion.
- Bigger model barely helps (+0.018); per-step recalibration DOESN'T transfer (and
  calibration is banned anyway). Margin separation is narrowest at hist 0
  (correct-median 1.5 vs 3.4 at 10+; wrong flat ~0.5-1.0).
- Figures (legacy location): figures/firststep_*.png, figures/gap_by_histlen.png.

## Results

**2026-07-07 · E1 RUNNING** — bge-m3 full-FT, serialize v1, @1024, epochs 3, lr 2e-5, bs4×ga4.
Both arms evaluate on the SAME 1,807 zero-history (first-step) val slice.

- **task0 `--zero_history`** (7,193 real first-steps) — **DONE: val_macro_f1 = 0.4283 uncal**
  (`output/pat/ft_results_firststep_0.csv`; calibrated col 0.494 ignored per no-calibration).
  vs generalist baseline **0.555 → Δ −0.127**. The zero-history *specialist* is **much worse**
  than the generalist — 7k first-step samples badly underfit a full-FT bge-m3. Preliminary read:
  a dedicated first-step specialist does not reach the generalist; first-step weakness is not
  fixable by specialization (it *hurts*).
- **task1 `--strip_history`** (all 56k, history stripped, `turn=` kept) — RUNNING (~67%, GPU 3).
  Controls for data quantity: if it recovers toward ~0.55 → task0's shortfall was small-data,
  not intrinsic; if it also stays low → the first-step mapping itself is hard. Full A/B bar
  figure + read-out on completion.
