# Branch: serialization — input-format gains on the best backbone
Git branch: `research/serialization` · Experiments: E2 (richargs × qwen3) — GATED on E8b: run only as attribution diagnostic if E8b disappoints (E8b subsumes the serialization axis)
Baseline: qwen3 v1@512 = 0.7682 uncal.

## Prior findings
- On bge-m3: richmeta +0.0117 uncal (0.7498→0.7615); richargs = exact null vs richmeta (+0.00002, trainer_state-verified); all info-REMOVAL variants regress (nometa −0.022, leanact −0.028, combo −0.040).
- Hypothesis: serialization and backbone gains are ~orthogonal → qwen3@richmeta ≈ 0.78.

## Results
(append here)
