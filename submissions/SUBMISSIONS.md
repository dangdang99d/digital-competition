# Submission tracker

Catalog of built submission zips (each `submissions/*.zip` contains the trained model
weights) **plus pending candidates** — so this doubles as "what still needs to be submitted."

**Columns to record:** local validation (uncal macro-F1) · file size · inference time (LB) ·
test performance (LB). Cells marked `—` are **to be recorded** (fill LB values from the DACON
leaderboard). `?` = expected to exist but not yet found in our notes.

⚠️ **Calibration:** all 8 existing zips ship a `logit_bias.json` (**CALIBRATED**, pre the
2026-07-07 no-calibration decision). New submissions (E8a/E8b, …) are **UNCALIBRATED** by policy —
their local-val numbers are raw-logit argmax and are NOT directly comparable to the calibrated zips'
historical CV.

## Built (already submitted)

| # | File | Backbone | Serialize | Pruned | Calib | Local val (uncal) | Size | Infer (LB) | Test (LB) | Notes |
|---|------|----------|-----------|:------:|:-----:|-------------------|-----:|-----------:|----------:|-------|
| 1 | submit_names_single.zip | granite-311m (fold1) | names (filenames-only) | no | yes | — (fold-CV ~0.7725*) | 515M | 5:10 | **0.77427** | ⭐ **LB SOTA**; all-data; fold-ensemble-ready (ships only 1 fold) |
| 2 | submit_0703_qwen3_pruned.zip | qwen3-0.6b | v1 | yes | yes | 0.7682 | 824M | 9:06 | ? | champion backbone (0.7752 cal); fp16 near 10:00 budget cliff |
| 3 | submit_0706_bgem3_richargs.zip | bge-m3 | richargs | yes | yes | — | 655M | — | 0.76274 | richargs **>** richmeta on LB despite byte-identical CV |
| 4 | submit_0706_bgem3_richmeta.zip | bge-m3 | richmeta | yes | yes | — | 655M | — | 0.76053 | |
| 5 | submit_0705_bgem3_full.zip | bge-m3 (hist0) | v1 | yes | yes | — | 655M | — | ? | full-data |
| 6 | submit_0705_bgem3_v2.zip | bge-m3 (v2) | v1 | yes | yes | — | 655M | — | ? | |
| 7 | submit_0705_bgem3_prune.zip | bge-m3 **12L** (depth-prune) | v1 | yes | yes | — | 388M | 2:29 | ? | depth + vocab pruned → smallest / fastest |
| 8 | submit_0703_bgem3_pruned.zip | bge-m3 | v1 | yes | yes | — | 655M | — | ? | early bge baseline |

\* fold-CV proxy per `combine/results.md` (may be calibrated); verify.

## Pending candidates (to build / submit)

| # | Candidate | Backbone | Recipe | Calib | Local val (uncal) | Size | Infer (LB) | Test (LB) | Status |
|---|-----------|----------|--------|:-----:|-------------------|-----:|-----------:|----------:|--------|
| E8a | granite + richargs + full_data | granite-311m | richargs · full-data | **no** | **0.7706** (Δ +0.0086 vs champ-qwen3 on 3.5k slice) | — | ~5 min class | — | 🔨 **ZIP TO BUILD** → submit |
| E8b | qwen3 + richargs + full_data | qwen3-0.6b | richargs · full-data | **no** | training (~2h) | — | ~9 min (watch budget) | — | 🏃 training |

Statuses: 🔨 build zip · ⬆ ready to upload · ✅ submitted · 🏃 still training
