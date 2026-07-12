# Branch: stacking (E29) — learned head over the trio ensemble

**STATUS: ❌ CLOSED (2026-07-12) — arm ① failed the gate on honest data.** Head CV ran as a
local experiment (user GO); the no-calibration ruling was never needed — nothing beat uniform,
so there is nothing to ship. User overrode the gate for ② and ④ (running 2026-07-12 late).

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
| ③ | per-row gating | probs (42) → member weights | small MLP → softmax weights | only if ② stalls; global weights already ≈ uniform (w424 LB 0.78709 vs 0.78719) |
| ④ | gated fusion (user 2026-07-12) | embeddings (3×768) → member weights | linear gate + softmax, trained through the blend (NLL) | ②'s rich input + ③'s constrained output — can only BLEND member opinions, never invent a class |

- CV protocol: session-grouped folds REUSED from E30 (a head trained on fold-k rows is evaluated
  on rows whose OOF predictions came from disjoint sessions).
- Final read: best head → one LB slot vs the uniform trio 0.78719. Slice/CV never ranks the
  final answer (E8/E26/E27 lesson — LB judges).

## Results

Substrate: honest E30 OOF caches (70k rows; member solo mF1 0.7644–0.7668, matching the
fold-models' own logs). **Uniform-mean null = 0.7745** — the +0.008 ensemble lift over solos
reproduces the LB gain (+0.0079) on honest OOF, independently validating E26.

Two objectives per arm — likelihood (the first pass; optimizes accuracy-like CE/NLL) vs
F1-oriented (user 2026-07-12: balanced class weights for ①/②, differentiable soft-F1
through the blend for ③/④). Gate to matter: ≥ +0.002 over the null.

| arm (input → output) | likelihood objective | F1-oriented objective |
|---|---|---|
| ① probs → class (logreg) | −0.0026…−0.0017, 0/5 folds ❌ | **+0.0007, 4/5 folds** ❌ under gate — sign FLIPPED, objective mattered ≈ +0.003 |
| ② embeddings → class (linear) | −0.0462, 0/5 ❌ | −0.0496, 0/5 ❌ — at 2304 dims capacity, not objective, is the killer (twins within 0.003); severe overfit despite 70k rows |
| ③ probs → member weights (gate net) | not run | −0.0039, 0/5 ❌ |
| ④ embeddings → member weights (gate net) | −0.0059, 0/5 ❌ — weights collapse to e25c (0.77–0.92/fold): NLL provably drifts to the best-calibrated member | −0.0090, 0/5 ❌ — soft-F1 is batch-noisy for rare classes; worst cell of the matrix |

**FINAL VERDICT — E29 ❌ CLOSED (2026-07-13). Uniform mean survives all nine attempts.**
Matrix complete: 4 arms × 2 objectives (+ the C-sweeps), best cell +0.0007 (①-balanced),
everything else negative. The pattern decomposes cleanly: at low capacity (①, 42 dims) the
OBJECTIVE is what matters — the user-diagnosed accuracy-vs-macro-F1 mismatch was worth ≈+0.003
and flipped the sign — but fixing it only buys parity; at high capacity (②, 2304 dims) or
under output constraints (③/④ gating) nothing helps: NLL chases calibration (④ collapses to
e25c), soft-F1 is batch-noisy on rare classes, balanced weighting overfits rare-class regions.
The mean's edge is structural: it preserves error-diversity across members at zero fitted
parameters, and every trained objective spends that diversity to optimize its proxy. Combined
with E30 (the same diversity isn't distillable) the ensemble story closes fully consistent:
**diverse errors + uniform averaging at inference is the whole trick, and it can be neither
learned-past nor compressed-away.** The no-calibration ruling was never needed — nothing
produced a shippable head.

## Notes

- Prior evidence stack (why expectations are modest): w424 global weights ≈ uniform on LB ·
  3.5k fitted combiners all failed honest CV · E27 flip rule (hand-built special case of ①)
  LB-neutral. The one positive prior: cross-model features broke the MSP ceiling (E27 assessor
  0.8655 vs 0.8425) — member disagreement demonstrably carries signal; the question is whether
  a head converts it to macro-F1 better than plain averaging.
