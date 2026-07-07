# Branch: reasoning — reasoning-FT backbone + classification head
Git branch: `research/reasoning` · Experiments: E5 — ✋ ON HOLD (user decision 2026-07-07): needs further analysis/planning before any run; do not dispatch
Baseline: qwen3 v1@512 plain full-FT = 0.7682 uncal (the identical-recipe control).

## Prior findings
- Labels weakly semantic: zero/few-shot reasoning caps ~0.28; budget kills generative
  inference (~300× over 50 samples/s) — hence train-time-only reasoning, head-only inference.
- Every aux objective so far hurt (supcon −0.007) or was null; skeptical prior.

## Results
(append here)
