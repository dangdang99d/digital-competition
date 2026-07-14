# Knowledge distillation — teacher(s) → smaller student

**Method:** train a single student on the (soft) outputs of a teacher / ensemble. Collapses
N models → 1 (both size and speed). Index: [results.md](results.md).

## Status: CLOSED for the trio→single case
- **E26-1B ❌:** 4 students 0.7635–0.7728 < champion 0.7803 — in-sample-harvested teachers ≈
  noisy labels, and KD costs the LS gain.
- **E30 OOF-teacher retry ❌:** honest session-grouped OOF teacher (fixes the in-sample
  harvest); best student still < champion. Trio's edge is **inference-time averaging**, not
  distillable into one 311m. Detail: [ensemble/results_e30.md](../ensemble/results_e30.md),
  memory [[distillation-not-viable]].

## Read-out
Distillation is not a live compression lever here — the ensemble advantage doesn't compress
into a single student. If a single strong model (e.g. the E38 AWP t031, LB 0.79300) is the
deployment target, there's no ensemble to distill anyway. Revisit only with a materially
different teacher/regime.

## Related
- Minitron-style distill-**recovery** (KD from the unpruned parent to recover a pruned child)
  is a different use — a possible add-on to [depth-pruning.md](depth-pruning.md) /
  [low-rank-factorization.md](low-rank-factorization.md) recovery, not ensemble→single. Untried.
