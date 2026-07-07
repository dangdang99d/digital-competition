# Branch: deferral — route low-confidence samples to a better source
Git branch: `research/deferral` · Experiments: E6 (two-model per-class-τ fallback robustness). E7 session-lookup REMOVED — competition-rules risk, do not attempt (user decision 2026-07-07)
Baseline: single-model champion qwen3 = 0.7682 uncal.

## Prior findings
- Margin (top1−top2, uncal) AUROC→correctness: hist0 0.832 / qwen3 0.848; wrong
  predictions pile at gap<1. Flip-to-top2 dead; pair-flip dead; higher-gap arbitration +0.001.
- Best: qwen3→hist0 fallback, per-class τ: 0.7732 held-out (+0.0049). History-conditioning
  adds nothing on top of class (class mix absorbs it). Oracle two-model ceiling 0.810.
- ~~Session-lookup~~: 🚫 DO NOT USE in any experiment or submission — potential
  competition-rules violation (user decision 2026-07-07). Data property stays noted
  in memory as background only.
- Figures (legacy): figures/logit_gap_*.png, figures/gap_by_histlen.png.

## Results

### E6 — Two-model per-class-τ fallback: ROBUSTNESS — **VERDICT: FAILS** (gain is slice-specific; recommend dropping the 2× fallback)
`analysis/gap_robustness.py` · uncalibrated · honest (τ fit on data disjoint from each eval slice) · baseline = single-model qwen3 0.7682.
Fallback = qwen3 top-1, but where qwen3's per-class gap < τ_c fall back to hist0's argmax; per-class τ_c greedy-fit (the champion R4' recipe from `gap_fix.py`).

**Overall (honest 2-fold, reproduction):** qwen3 base 0.7682 → fallback **0.7729 (+0.0047)**. Reproduces the prior +0.0049/0.7732 claim (fold seed differs by a hair). This is the number the fallback was proposed on.

**Robustness slice 1 — by generator** (fit τ on ONE generator's val subset, eval on the OTHER; generator parsed from `id` = `sess_<gen>_…`; val = sim 12966 / au 1034):

| held-out slice | n | qwen3 slice base | fallback | Δ vs slice base | Δ vs global 0.7682 |
|---|---|---|---|---|---|
| **au** (τ fit on sim) | 1034 | 0.8856 | 0.8823 | **−0.0033** | +0.1141 |
| **sim** (τ fit on au) | 12966 | 0.7570 | 0.7594 | +0.0024 | −0.0088 |

The τ tuned on sim **does not transfer to the held-out au generator — the fallback is net-NEGATIVE there (−0.0033).** au is the rare generator (~7%) whose test-mix is unknown, i.e. exactly the distribution-shift case robustness has to survive. (au's absolute F1 is high only because au is an easier slice, not because the fallback helps.)

**Robustness slice 2 — by step / history length** (honest 2-fold for τ within each stratum; step parsed from `id` = `…-step_NN`; step 1 = zero history):

| stratum | n | qwen3 slice base | fallback | Δ vs slice base | Δ vs global 0.7682 |
|---|---|---|---|---|---|
| first-step (step 1, zero-history) | 1807 | 0.5769 | 0.5753 | **−0.0016** | −0.1929 |
| later steps (step ≥ 2) | 12193 | 0.7720 | 0.7740 | +0.0020 | +0.0058 |

**On first-step / zero-history the fallback is net-NEGATIVE (−0.0016)** and far below 0.7682 (a hard slice regardless). The +0.0047 overall gain lives entirely in the sim-heavy, later-step majority regime.

**Verdict — FAILS.** Under 2-fold honesty the fallback beats single-model qwen3 on only 2 of the 4 robustness slices (sim +0.0024, later +0.0020) and **hurts on the two stress tests that matter — cross-generator transfer to held-out au (−0.0033) and zero-history first-step (−0.0016).** By the literal "beats 0.7682 on every slice" bar it also fails (held-out sim 0.7594 and first-step 0.5753 are below 0.7682). The +0.0047 headline is a majority-regime artifact, not a robust gain. Recommend **dropping the 2× fallback** as a submission candidate: it does not robustly beat single-model qwen3 and it doubles inference cost (needs both models' weights).

Figure: `figures/e6_robustness.png` (grouped bars, fallback vs single-model qwen3 across all slices; red dashed = global 0.7682).
