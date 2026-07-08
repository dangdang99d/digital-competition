# Branch: reasoning — reasoning-FT backbone + classification head
Git branch: `research/reasoning` · Baseline: **qwen3 v1@512 plain full-FT = 0.7682 uncal** (identical-recipe control).

## Summary
Legend: ✋ on hold.

| Exp | Experiment | Status | Verdict |
|---|---|:--:|---|
| E5 | reasoning-FT backbone + classification head | ✋ | ON HOLD (user decision) — not dispatched |

## E5 — reasoning-FT → head
- **What:** train with rationales, attach a classification head, then do head-only fast inference (1 forward pass, budget-safe).
- **Baseline:** qwen3 full-FT = 0.7682.
- **Change:** rationale-augmented training + linear classification head.
- **Result:** — **not run**.
- **Verdict:** ✋ **ON HOLD** (user decision 2026-07-07) — needs analysis/planning first. Skeptical prior: labels are weakly semantic (zero/few-shot reasoning caps ~0.28), generative inference blows the budget, and every aux objective so far hurt or was null (supcon −0.007).
