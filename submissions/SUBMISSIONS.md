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

**Two independent kinds of pruning — kept in separate columns:**
- **Vocab-prune** = drop unused tokenizer/embedding rows (via `remap.npy`) — shrinks the **embedding table only**, transformer body untouched. qwen3/bge get it; granite ships the full vocab.
- **Model-prune** = shrink the **transformer body** — depth-prune (drop layers) or FFN-factor (low-rank FFN).

| # | File | Backbone | Serialize | Vocab-prune | Model-prune | Calib | Local val (uncal) | Size | Infer (LB) | Test (LB) | Notes |
|---|------|----------|-----------|:-----------:|:-----------:|:-----:|-------------------|-----:|-----------:|----------:|-------|
| 1 | submit_names_single.zip | granite-311m (fold1) | names (filenames-only) | no | no | yes | — (fold-CV ~0.7725*) | 515M | 5:10 | 0.77427 | prior LB SOTA (now #3); all-data; fold-ensemble-ready (ships only 1 fold) |
| 2 | submit_0703_qwen3_pruned.zip | qwen3-0.6b | v1 | yes | no | yes | 0.7682 | 824M | 9:06 | ? | champion backbone (0.7752 cal); fp16 near 10:00 budget cliff |
| 3 | submit_0706_bgem3_richargs.zip | bge-m3 | richargs | yes | no | yes | — | 655M | — | 0.76274 | richargs **>** richmeta on LB despite byte-identical CV |
| 4 | submit_0706_bgem3_richmeta.zip | bge-m3 | richmeta | yes | no | yes | — | 655M | — | 0.76053 | |
| 5 | submit_0705_bgem3_full.zip | bge-m3 (hist0) | v1 | yes | no | yes | — | 655M | — | ? | full-data |
| 6 | submit_0705_bgem3_v2.zip | bge-m3 (v2) | v1 | yes | no | yes | — | 655M | — | ? | |
| 7 | submit_0705_bgem3_prune.zip | bge-m3 | v1 | yes | **depth 24→12** | yes | — | 388M | 2:29 | ? | vocab + depth pruned → smallest / fastest |
| 8 | submit_0703_bgem3_pruned.zip | bge-m3 | v1 | yes | no | yes | — | 655M | — | ? | early bge baseline |
| 9 | submit_0707_qwen3_ls.zip | qwen3-0.6b (E8b+LS) | richargs | yes | no | **no** | 0.7659 (full_data) | 830M | **9:18** | **0.77921** | 🥇 **NEW LB SOTA** (submitted 07-08). fp16, richargs vocab-prune verified. 9:18 under the 10-min cliff but tight |
| 10 | submit_0707_granite_ls.zip | granite-311m (E8a+LS) | richargs | no | no | **no** | 0.7803 (full_data) | 846M (fp32‡) | **5:06** | **0.77738** | 🥈 #2 LB (submitted 07-08). granite's full_data-CV lead did NOT hold on LB — qwen3_ls wins. ‡ships fp32; fp16 twin `submit_0707_granite_ls_fp16.zip` (539M, parity PASS) available |
| 11 | submit_0708_granite_tta_lr1e3.zip | granite-311m (E8a+LS **+TTA** lr1e-3) | richargs | no | no | **no** | 0.7737 (clean slice, −0.0034 vs no-TTA) | 846M | **7:43** | **0.77931** | 🥇 **NEW LB SOTA** (07-08). Test-time adaptation (SHOT/IM entropy-min on 45 encoder LayerNorm affines, 2048 test rows, fp32) on the granite_ls champion. **+0.00193 vs granite_ls 0.77738**; edges past qwen3_ls 0.77921 (+0.0001). **moderate lr = sweet spot** |
| 12 | submit_0708_granite_tta_lr2e4.zip | granite-311m (E8a+LS **+TTA** lr2e-4) | richargs | no | no | **no** | 0.7787 (clean, −0.0006) | 846M | **7:51** | **0.77871** | gentle TTA: **+0.00133 vs granite_ls**. Under-corrects relative to lr1e-3 (helps, but less) |
| 13 | submit_0708_granite_tta_lr3e3.zip | granite-311m (E8a+LS **+TTA** lr3e-3) | richargs | no | no | **no** | 0.7659 (clean, −0.0094) | 846M | **7:44** | **0.77300** | aggressive TTA: **−0.00438 vs granite_ls** — over-sharpens/overshoots. TTA has an optimum; "more" is worse |

\* fold-CV proxy per `combine/results.md` (may be calibrated); verify.
† full_data CV = eval on a held-out slice, not the 14k val — high-variance, doesn't rank-order the models; trust LB.

## Pending candidates (to build / submit)

| # | Candidate | Backbone | Recipe | Calib | Local val (uncal) | Size | Infer (LB) | Test (LB) | Status |
|---|-----------|----------|--------|:-----:|-------------------|-----:|-----------:|----------:|--------|
| E8a+LS | granite + richargs + full_data + LS | granite-311m | richargs · full-data · LS · no prune (full granite) | **no** | 0.7803 (3.5k slice) | 846M | 5:06 | **0.77738** | ✅ **SUBMITTED** → row 10 (LB 0.77738) |
| E8b+LS | qwen3 + richargs + full_data + LS | qwen3-0.6b | richargs · full-data · LS · vocab-prune | **no** | 0.7659 (3.5k slice) | 830M | 9:18 | **0.77921** | ✅ **SUBMITTED** → row 9 (🥇 LB 0.77921, new SOTA) |
| E16z | qwen3 **depth-14** (half) + recovery | qwen3-0.6b | richargs · full-data · vocab-prune + model-prune (depth 28→14) | **no** | **0.7638** (full_data) | **443M** | **~½ of qwen3 (unmeasured)** | — | ✅ **BUILT — ready to upload**: `submit_0708_qwen3_depth14.zip` (443M, fp16, parity 0/3000, remap reused from qwen3_ls). Value: fast candidate for the 10% speed score — but inference time is only knowable from an actual submission. Recovered from E8b (no LS), so expect LB a bit under qwen3_ls 0.77921 |
| E4z | qwen3 **FFN-factored r512** + recovery | qwen3-0.6b | richargs · full-data · vocab-prune + model-prune (FFN-factor r512) | **no** | **0.7688** | ~2.0GB ckpt | ~qwen3 | — | 📦 **TRAINED, READY TO PACKAGE** — `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e4_ffn_r512_recover_e8b/checkpoint-4157`. **smaller AND better** than uncompressed qwen3 (0.7643). ⚠️ needs `load_factored_model` hook wired into `script.py` (not standard from_pretrained) — smoke-test with user |

Statuses: 🔨 build zip · ⬆ ready to upload · ✅ submitted · 🏃 still training · 📦 trained, ready to package

**LB note (2026-07-08):** both LS zips submitted (rows 9–10), both beat prior SOTA 0.77427. **CV→LB reversal:** granite led
on full_data CV (0.7803 vs qwen3 0.7659) but qwen3_ls won LB (0.77921 vs 0.77738) — trust LB, not full_data CV. Speed: qwen3
9:18 (tight), granite 5:06 (margin); if the 10% speed sub-score matters, granite (or compressed-qwen3 E16z/E4z) is the safer fast entry.

**fp16 rebuild (2026-07-08):** the submitted `submit_0707_granite_ls.zip` (846M) ships **fp32** (build skipped `.half()`);
it ran fine at 5:06 so no re-submit needed. A parity-verified fp16 twin **`submit_0707_granite_ls_fp16.zip` (539M)** exists as
the equivalent smaller upload (14k-val parity: 13997/14000 agree, ΔF1 −0.00006 → PASS). Original fp32 zip kept (do not overwrite).

**TTA experiment (2026-07-08) — rows 11–13, TTA WORKS:** unsupervised test-time adaptation bolted onto the granite_ls
champion's inference (SHOT/information-maximization entropy-min — sharpen + anti-collapse diversity term — on granite's
**45 encoder LayerNorm affines only**, head frozen; a bounded 2048-row sample of the test set; fp32, gradient-checkpointed;
prediction pass unchanged from the champion). **There IS exploitable test-shift.** vs granite_ls baseline **0.77738**:
gentle `lr2e-4` **+0.00133** (0.77871), moderate `lr1e-3` **+0.00193** (0.77931), aggressive `lr3e-3` **−0.00438** (0.77300).
Best = **lr1e-3 → 0.77931, a hair past the prior LB SOTA qwen3_ls 0.77921**. Optimum is intermediate: gentle under-corrects,
aggressive overshoots (over-sharpens confidence past the shift correction). **Local vs LB divergence (key lesson):** the clean
no-shift held-out slice predicted all three NEGATIVE (−0.0006 / −0.0034 / −0.0094); the real test flipped gentle+moderate
POSITIVE — the local clean slice is only the *no-shift floor*, TTA's real payoff lives in test-shift we can't see locally, so
trust the LB. **Timing:** ~7:43–7:51 — TTA added **~2:40** on DACON HW (vs a ~40s local 3090 estimate); safe on granite's
5:06 base but this validates NOT running TTA on qwen3_ls (9:18 + ~2:40 → over the 10-min cliff). `TTA_ENABLE=0` reproduces the
champion byte-for-byte. Build: reused the champion model verbatim, swapped only `script.py`; harness `experiments/misclf-detection/tta_eval.py`.
