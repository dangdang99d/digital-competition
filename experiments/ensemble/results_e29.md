# Branch: stacking (E29) — learned head over the trio ensemble

**STATUS: ⛔ GATED (nothing run).** Two gates: (a) **E30 phase-0 OOF caches**
([results_e30.md](results_e30.md) — running 2026-07-12), (b) **user ruling on the
no-calibration invariant** — a trained head reshapes class probabilities; uniform mean was
explicitly exempted (2026-07-09), a learned combiner was not. No build until both clear.

## Objective

Replace the trio's uniform softmax mean (LB 0.78719) with a TRAINED head (user ask 2026-07-10:
"ML model on top instead of hand-built"). Members are FIXED = the LB-certified trio
(is3 · aum06 · e25c); the head ships alongside them (KBs → packaging/time unchanged, 6:52/836M).

## Why the training data must be E30's OOF (not the 3.5k slice)

At n=3,500 every fitted combiner LOST its honest 2-fold CV while "winning" its full-slice fit
(E26 shoot-out: stacking 0.7776/0.7799 vs uniform 0.7840/0.7851 — textbook overfit). The E30
harvest gives 70k leakage-free rows (20× the data) with both probs and penultimate embeddings
from all three member recipes — the first setting where a learned head gets a fair trial.

## Arms — strict gate order (user-picked options 1→2→3)

| # | arm | input (per row) | model | promote/kill rule |
|---|---|---|---|---|
| ① | prob-stacker (GATEKEEPER) | concat member OOF probs (3×14) | L2 logreg (GBM only if logreg wins) | ≥ **+0.002** honest 5-fold CV over uniform-mean null, else **E29 CLOSES** (② and ③ inherit the same data limits with more capacity) |
| ② | embedding-fusion head | concat penultimate embeddings (3×768) | linear / small MLP → 14 | best CV vs ① and uniform; members ship headless — the fusion head becomes the classifier |
| ③ | per-row gating | probs (or input emb) → member weights | small MLP → softmax weights | only if ② stalls; global weights already ≈ uniform (w424 LB 0.78709 vs 0.78719) |

- CV protocol: session-grouped folds REUSED from E30 (a head trained on fold-k rows is evaluated
  on rows whose OOF predictions came from disjoint sessions).
- Final read: best head → one LB slot vs the uniform trio 0.78719. Slice/CV never ranks the
  final answer (E8/E26/E27 lesson — LB judges).

## Results — 🔲 pending gates

| arm | honest CV (Δ vs uniform null) | LB | verdict |
|---|---|---|---|
| ① logreg probs | — | — | — |
| ② fusion head | — | — | — |
| ③ gating | — | — | — |

## Notes

- Prior evidence stack (why expectations are modest): w424 global weights ≈ uniform on LB ·
  3.5k fitted combiners all failed honest CV · E27 flip rule (hand-built special case of ①)
  LB-neutral. The one positive prior: cross-model features broke the MSP ceiling (E27 assessor
  0.8655 vs 0.8425) — member disagreement demonstrably carries signal; the question is whether
  a head converts it to macro-F1 better than plain averaging.
