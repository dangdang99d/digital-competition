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

⚠️ **Filename limit (user, 2026-07-10):** DACON caps the submission filename at **32 characters
including `.zip`** — several past names exceeded it and needed manual pruning at upload. Name new
zips ≤32 chars (aim ≤28): shorten the variant tag, keep `submit_MMDD_`.

## Eval-server batch sizes (T4 16GB — measured VRAM, 2026-07-10)

Peak allocator bytes measured on a 3090 (allocation is GPU-independent → valid for T4);
worst case = every row padded to 512. T4 budget ≈ **13.5 GiB** (16GB − context − margin).
Old zips shipped **bs=64** — far below the ceiling. Details/sweep tables: `experiments/ensemble/results_e26.md`.

| backbone | dtype | **max safe bs @512** | peak there | ship setting |
|---|---|---|---|---|
| granite-311m | fp16 | **512** (4.48 GiB — memory never binds) | 4.48 GiB | **bs 256** (E26 zips; `ENS_BS` env override) |
| granite-311m | fp32 | 512 (8.94 GiB) | 8.94 GiB | bs 256 fine |
| qwen3-0.6b | fp16 | **128** (9.99 GiB) — bs 256 = 18.9 GiB OVER | 9.99 GiB | bs 128 (2× the shipped 64; likely compute-bound anyway) |
| qwen3-0.6b | fp32 | 64 only (10.85 GiB, tight) | 10.85 GiB | avoid fp32 qwen3 |
| qwen3 depth-14 | fp16 | ~256 (est. ~½ of full qwen3's activations) | est ~10 GiB | unmeasured — verify before shipping |

Also: pre-tokenize ONCE + length-sorted batching (3 vCPU on the server — per-batch tokenization
in the loop wastes CPU time; the E26 script is the reference implementation).

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
| 14 | submit_0710_ens2_is2_estack.zip | granite-311m ×2 (E26-A ensemble) | richargs | yes (union 54,688) | no | **no** | 0.7832 ens (3.5k slice) | 565M | **4:30** | **0.78548** | 🥈 E26-A pair (is2+estack, uniform prob mean, fp16 bs256). Was SOTA for ~4 h; pair FASTER than single fp32 champion (5:06) |
| 15 | submit_0710_ens2_is3_richmeta.zip | granite-311m ×2 (E26-B ensemble) | richargs+richmeta | yes (union) | no | **no** | 0.7840 ens | 559M | **4:46** | **0.78498** | E26-B cross-serialization pair; slice preferred it over A, LB reversed — slice mis-ranks ensembles |
| 16 | submit_0710_s43_full.zip | granite-311m (seed-43 reroll) | richargs | no | no | **no** | 0.7758 own-split (0.7955 seed-42 slice = LEAK) | 832M | **5:12** | **0.77427** | leak probe: LB confirms 0.7955 was memorization; seed reroll < champion |
| 17 | submit_0710_ens3_is3_aum06_richmeta.zip | granite-311m ×3 (E26-C ensemble) | richargs×2+richmeta | yes (union) | no | **no** | 0.7850 ens | 836M | **6:52** | **0.78719** | 🥈 LB SOTA 07-10→07-13 (now #2, beaten by E28-trio row 22 0.78780). Trio = direct-package ceiling (4th member breaks both caps); marginal member ≈2:22 |
| 18 | submit_0710_ens3_w424.zip | granite-311m ×3 (E26 trio, weighted variant) | richargs×2+richmeta | yes | no | **no** | — (built outside main session; presumed 0.4/0.2/0.4 member weights — confirm composition) | ~836M | **6:49** | **0.78709** | weighted trio ≈ uniform trio (0.78719, −0.0001) — confirms the combiner shoot-out: weights don't beat uniform; uniform stays champion |
| 19 | submit_0710_flip.zip | granite-311m (E8a+LS + **E27 flip rule**) | richargs | no | no | **no** | 0.7813 (e8a 3.5k slice, +0.0012) | 846M | **5:06** | **0.77727** | E27 LB probe (submitted 07-10 19:05): 8-cell rank-2 swap on low-MSP (r1,r2,decile) cells. **−0.00011 vs champion 0.77738 → NEUTRAL on LB; the +0.0012 slice gain did NOT transfer.** Zero inference-time cost (5:06 = champion). E27 detect-and-fix line CLOSED |
| 20 | submit_0713_g_t043.zip | granite-311m (E28 t043, **all_data**) | richargs | no | no | **no** | fold-0 0.7724 (search); no promotion CV | 540M | ~5:0x | **0.75934** | ❌ E28 winner shipped via `--all_data` = **last-epoch overfit tail** (top configs peak @ep3, all_data saved ep5). −0.018 vs champion. **SUPERSEDED by row 21.** Lesson: best-epoch objective ⇒ promote best-epoch (`--full_data`), never blind last-epoch |
| 21 | submit_0713_g_t043fd.zip | granite-311m (E28 t043, **full_data**) | richargs | no | no | **no** | 3.5k slice 0.7705; fold-0 **0.7724** | 555M | **5:04** | **0.78155** | E28 single-model SOTA at submission time (+0.0042 vs granite_ls 0.77738, +0.0022 vs TTA 0.77931, > qwen3_ls 0.77921). E28 optuna search + `--full_data` best-epoch. Validates the leak-free fold over the 3.5k slice. [optuna/results.md](../experiments/optuna/results.md) |
| 22 | submit_0713_e28_trio.zip | granite-311m ×3 (E28 trio t019+t023+t043) | richargs ×3 | yes (union 54,688) | no | **no** | fold-0 14k ens **0.7774** (+0.0055 vs best single) | 842M | **6:39** | **0.78780** | 🥇 **NEW LB SOTA (07-13)** (+0.00061 vs E26-C 0.78719). E28 optuna trio, uniform softmax mean, fp16 bs256. **+0.00625 over its best single t043fd 0.78155 — matches the +0.0055 fold screen** (14k session-fold-0, leak-free; screen was reliable). Diversity = lr-spread only, yet edges E26-C. parity 0/1000 all members. [optuna/results.md](../experiments/optuna/results.md) |

\* fold-CV proxy per `combine/results.md` (may be calibrated); verify.
† full_data CV = eval on a held-out slice, not the 14k val — high-variance, doesn't rank-order the models; trust LB.

## Pending candidates (to build / submit)

| # | Candidate | Backbone | Recipe | Calib | Local val (uncal) | Size | Infer (LB) | Test (LB) | Status |
|---|-----------|----------|--------|:-----:|-------------------|-----:|-----------:|----------:|--------|
| E16z | qwen3 **depth-14** (half) + recovery | qwen3-0.6b | richargs · full-data · vocab-prune + model-prune (depth 28→14) | **no** | **0.7638** (full_data) | **443M** | **~½ of qwen3 (unmeasured)** | — | ✅ **BUILT — ready to upload**: `submit_0708_qwen3_depth14.zip` (443M, fp16, parity 0/3000, remap reused from qwen3_ls). Fast candidate for the 10% speed score. Recovered from E8b (no LS), so expect LB a bit under qwen3_ls 0.77921 |
| E4z | qwen3 **FFN-factored r512** + recovery | qwen3-0.6b | richargs · full-data · vocab-prune + model-prune (FFN-factor r512) | **no** | **0.7688** | ~2.0GB ckpt | ~qwen3 | — | 📦 **TRAINED, READY TO PACKAGE** — `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e4_ffn_r512_recover_e8b/checkpoint-4157`. **smaller AND better** than uncompressed qwen3 (0.7643). ⚠️ needs `load_factored_model` hook wired into `script.py` (not standard from_pretrained) — smoke-test with user |
| E28-t017fd | t017 config (low-ε **0.022**, lr 4.46e-5, 4ep, eb16) `--full_data` — **slice-CV hedge** | granite-311m | richargs · full_data · LS ε 0.022 · fp16 full-vocab | **no** | 3.5k slice **0.7750** (tops all E28 on slice); fold-0 0.7682 (weakest on fold) | **536M** | ~5:0x | — | ✅ **BUILT 07-13 — ready to upload**: `submit_0713_g_t017fd.zip`. ⚠️ fold vs slice DISAGREE (fold: t043>t017; slice: t017>t043). Submit to test whether the slice's t017-preference beats t043fd's LB 0.78155 |
| E28-t048fd | t048 config (lr 3.52e-5, 5ep, eb8, ε 0.174) `--full_data` — runner-up | granite-311m | richargs · full_data · LS ε 0.174 · fp16 full-vocab | **no** | 3.5k slice 0.7693; fold-0 0.7715 | ~555M | ~5:0x | — | ✅ **BUILT 07-13 — ready to upload**: `submit_0713_g_t048fd.zip` (CPU sanity 7/8). E28 diversity/insurance member |
| E28-t023fd | t023 config (lr 4.35e-5, 5ep, eb8, ε 0.116, low wd) `--full_data` — diverse | granite-311m | richargs · full_data · LS ε 0.116 · fp16 full-vocab | **no** | 3.5k slice 0.7697; fold-0 0.7709 | ~555M | ~5:0x | — | ✅ **BUILT 07-13 — ready to upload**: `submit_0713_g_t023fd.zip` (CPU sanity 7/8). E28 diversity member (in-basin, distinct ε/wd) |
| E34-C AWP | **AWP adversarial weight perturbation** (champion recipe + `--awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0`) | granite-311m | richargs · full_data · LS ε 0.1 · **+AWP** · fp16 full-vocab | **no** | full_data 3.5k slice **0.7804** = **+0.0071 vs its own from-scratch anchor 0.7733** (E34, vast; anchor sits below champion 0.7803 due to E19 fast-shape batching — the Δ is the real signal); matches documented champion 0.7803 | **601M** | **5:10** | **0.78557** | ✅ **SUBMITTED 07-13 → 🥇 LB 0.78557 = NEW SINGLE-MODEL SOTA** (+0.0040 vs prev best single E28 t043fd 0.78155; +0.0082 vs champion granite_ls 0.77738). A SINGLE model that **ties the 2-granite ensemble pairs** (pair A 0.78548) at 5:10 (< trio 6:52). Slice under-predicted again (0.7804 → 0.78557, +0.005). `submit_0713_awp.zip`. AWP confirmed on LB → prime ensemble-member candidate |

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
