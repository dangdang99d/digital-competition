# Branch: serialization — input-format gains on the best backbone
Git branch: `research/serialization` · Experiments: E2 (richargs × qwen3) — GATED on E8b: run only as attribution diagnostic if E8b disappoints (E8b subsumes the serialization axis)
Baseline: qwen3 v1@512 = 0.7682 uncal.

## Summary (at-a-glance)
Legend: ⛔ gated, not run. Baseline = qwen3 v1@512 0.7682 (uncal).

| Exp | Experiment | Status | Result | Verdict |
|---|---|:--:|---|---|
| E2 | richargs × qwen3 (single-axis ablation) | ⛔ | — | Gated on E8b — run ONLY as attribution diagnostic if E8b disappointed. E8b landed fine (0.7643) and the richargs axis is already in the champion recipe → **correctly skipped** |

## Prior findings
- On bge-m3: richmeta +0.0117 uncal (0.7498→0.7615); richargs = exact null vs richmeta (+0.00002, trainer_state-verified); all info-REMOVAL variants regress (nometa −0.022, leanact −0.028, combo −0.040).
- Hypothesis: serialization and backbone gains are ~orthogonal → qwen3@richmeta ≈ 0.78.

## Results
(append here)
