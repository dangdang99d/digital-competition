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

| 23 | submit_0714_awp_t031.zip | granite-311m (E38 AWP-optuna t031, **full_data** ep4) | richargs | no | no | **no** | 3.5k slice **0.7859**; fold-0 14k **0.7806** | 659M | **5:04** | **0.79300** | 🥇🥇 **NEW SINGLE-MODEL SOTA + CLEARS #12 (0.79282) by +0.00018.** Optuna-tuned AWP (lr3.5e-5·awp_γ2.0e-3·awp_lr2.3e-4·ep4). **+0.00743 vs E34 AWP 0.78557** (slice edge +0.0055 → LB +0.0074, over-transferred), **+0.00384 vs the whole 4-member team submission 0.78916.** Full-vocab fp16, raw argmax, mirrors E34 AWP zip. A SINGLE model beat the team ensemble. [optuna/results_e38.md](../experiments/optuna/results_e38.md) |

| 24 | submit_0714_elr_t031.zip | granite-311m (E42 ELR+AWP, t031 config, **full_data** ep4) | richargs | no | no | **no** | 3.5k slice **0.7856** | 659M | **5:04** | **0.79311** | **ELR-effect isolation test** (same t031 config as row 23, ELR vs no-ELR): **+0.00011 vs AWP t031 0.79300 = DEAD TIE** (LB noise ~1e-4). Confirms E22/E35 prior *on LB*: AWP+LS already strong, ELR adds nothing as a single (slice −0.0003 agreed). Marginal-best single, but value is only as a thin same-config ensemble ingredient. [noise-robust/results_e42_elr_awp.md](../experiments/noise-robust/results_e42_elr_awp.md) |
| 25 | submit_0714_qwen3_t011.zip | **qwen3-0.6B** (E41 AWP-optuna t011, **full_data** ep4) | richargs | **yes** (29.7k vocab, margin 25k) | no | **no** | 3.5k slice **0.7827**; fold-0 14k **0.7776** | 958M | **9:14** | **0.79130** | 🧪 **Cross-backbone probe — RESULT: loses to granite (−0.0017 vs t031 0.79300).** AWP-optuna qwen3 (lr1.17e-5·ls0.077·awp_γ2.3e-3·awp_lr1.15e-4·ep4). **The E8 slice→LB shift partially held but was too small:** CV gap vs granite was −0.0030 (fold) / −0.0032 (slice), LB gap only **−0.0017** → qwen3 climbed ~+0.0013 relative to granite on LB, not the ~+0.016 E8 implied — nowhere near enough to overtake. **qwen3 single confirmed weaker than granite AWP on the real LB too.** Residual value = ensemble diversity only, but at **9:14** a qwen3-containing ensemble busts the 10-min cliff. Vocab-pruned (full fp16 1.06GB over cap); parity 800/800 (100%) pruned≡full; raw argmax. [optuna/results_e41_qwen3.md](../experiments/optuna/results_e41_qwen3.md) |
| 26 | submit_0714_awp_nf4.zip | granite-311m (E38 AWP-optuna t031, **full_data** ep4) **+ nf4 4-bit quant** | richargs | no | **nf4 quant** | **no** | (same weights as row 23) | **500M** | **3:33** | **0.79307** | ⚡ **nf4 quantization of the champion (row 23) — FREE speedup, no accuracy cost.** **+0.00007 vs full-precision t031 0.79300 = dead tie** (LB noise), but inference **3:33 vs 5:04** (−30%) and size 500M vs 659M. 4-bit weight quant on the SOTA single is lossless on this task → the fast/small variant to ship if the 10% speed sub-score matters. Still clears #12 (0.79282). |
| 27 | submit_0714_drophd.zip | granite-311m (E47 **DropHead** p=0.1 + t031 AWP recipe, **full_data** ep5) | richargs | no | no | **no** | 14k-screen **0.7826** (56k); 3.5k slice **0.7839** | 629M | **4:55** | **0.78866** | ❌ **DropHead does NOT transfer — −0.00434 vs AWP t031 0.79300.** Won the honest 14k screen (+0.0044 vs anchor 0.7782, both 56k/6ep) but the full_data/LB reversed it — consistent with DropHead's stochastic-mask run-variance ~0.005 (wave-2: 8ep p0.1 0.7769 vs 6ep 0.7826) and the 3.5k slice (0.7839 < AWP 0.7856), which predicted it. Above old trio 0.78780, below the champion single. **AWP t031 stays SOTA.** Screen win was likely single-seed noise → a *controlled* Optuna (pinned epochs, multi-seed) is the only fair retest. [performance-boost/results_e47.md](../experiments/performance-boost/results_e47.md) |
| 28 | submit_0715_msd_s42.zip | granite-311m (E50 **MSD head** k5 p0.3 + t031 AWP recipe, **full_data** ep5/6) | richargs | no | no | **no** | 3.5k **0.7900** (top of the whole fleet) | 629M | **5:07** | **0.79121** | ❌ −0.0018 vs champion. Best 3.5k model of all 16 → LB *below* champion: the slice over-predicts regularizers, same as DropHead. |
| 29 | submit_0715_ramp_s42.zip | granite-311m (E49 **layer-ramp DropHead** p=0.1, **full_data** ep8/best5) | richargs | no | no | **no** | 3.5k 0.7843 | 629M | **5:16** | **0.78552** | ❌ **worst single** (−0.0075 vs champion). The E49 3.5k standout inverted hardest on LB → DropHead family closed. |
| 30 | submit_0715_awpseed3.zip | granite-311m **seed trio** = awpfd_s23+s17+s2 (t031 recipe, **full_data** ep4, 3 init_seeds) | richargs | **yes** (54,688; 292M/member) | no | **no** | members 3.5k 0.7884/0.7853/0.7846 | 909M | **6:42** | **0.79340** | 🥇 **first ensemble to beat the champion single** (+0.0004 vs 0.79300). Pure-seed members = highly correlated → only a small gain, but proved averaging > any single. |
| 31 | submit_0715_s23.zip | granite-311m awpfd_s23 single (best AWP seed by 3.5k) | richargs | no | no | **no** | 3.5k **0.7884** (#1 seed) | 629M | **5:05** | **0.79130** | ❌ **the key negative:** best-3.5k seed → near-worst LB single. **The 3.5k doesn't just add noise, it MIS-RANKS seeds** (t031 was mid-slice/top-LB) → never select members by slice score. |
| 32 | submit_0715_champtrio.zip | granite-311m **champion trio** = **t031** + awpfd_s17 + awpfd_s2 | richargs | **yes** (54,688) | no | **no** | members 3.5k 0.7856/0.7853/0.7846 | 909M | **6:35** | **0.79459** | 🏆🏆 **BEST OF THE CAMPAIGN — final selected submission.** Swapped the trio's LB-worst member (s23) for the champion: **+0.0012 vs seed trio, +0.0016 vs champion single.** Champion-anchored + decorrelated seeds. |
| 33 | submit_0715_s2.zip | granite-311m awpfd_s2 single | richargs | no | no | **no** | 3.5k 0.7846 | 629M | **5:10** | **0.78866** | ❌ weak single (−0.0043) — **yet it IMPROVES the champtrio.** Proves the ensemble runs on *decorrelation*, not member strength: a weak-but-decorrelated member lifts the mean. |
| 34 | submit_0715_s17.zip | granite-311m awpfd_s17 single | richargs | no | no | **no** | 3.5k 0.7853 (below champion's 0.7856) | 629M | **5:08** | **0.79365** | ✅ **NEW BEST SINGLE — beats champion t031 (0.79300) by +0.0007.** A seed genuinely stronger than the tuned champion, and the 3.5k ranked it *below* t031 → slice mis-ranking confirmed a 2nd time. |
| 35 | submit_0715_pair_t031_s17.zip | granite-311m **pair** = t031 + awpfd_s17 (the two strongest singles) | richargs | **yes** (54,688) | no | **no** | — | 617M | **4:35** | **0.79448** | 🥈 2-member of the best singles ≈ champtrio (−0.0001) at **4:35 vs 6:35** — near-identical accuracy, 2 min faster. Best accuracy/speed trade of the campaign. |
| 36 | submit_0715_t031_s17_s23.zip | granite-311m trio = t031 + s17 + **s23** (s2→s23 swap) | richargs | **yes** (54,688) | no | **no** | — | 909M | **6:44** | **0.79422** | ❌ **s23 as 3rd member HURTS** (−0.0037 vs champtrio; even below the 2-member pair 0.79448) despite being *stronger solo* than s2 (0.79130 vs 0.78866). **Decisive proof: correlated-strong < decorrelated-weak.** |

| 37 | submit_0715_awp_trio.zip | granite-311m **AWP trio** — ⚠️ built outside this session; exact member list unverified (not on local disk) | richargs | (trio-sized ⇒ likely yes) | no | **no** | — | — | **7:11** | **0.79242** | 🟡 −0.0022 vs champtrio 0.79459. Uploaded 09:53. A 3-member AWP ensemble that lands **below** the champion-anchored trio — consistent with the campaign rule: a trio of *correlated* AWP members ≈ the seed trio (0.79340) tier, while the champion anchor + decorrelated seeds is what reaches 0.7946. |
| — | submit_0715_champquad.zip | granite-311m **quad** = t031+s17+s2+**e25c_richmeta** (16,473-row prune, 4 members) | richargs **+ richmeta** (mixed) | **yes** (16,473; 4 members) | no | **no** | — | 977M | (est ~8:30) | **not uploaded** | 🔲 **BUILT + smoke-passed, never submitted** (slots ran out). The one untried lever: adds the *serialization-diverse* member (different input text ⇒ least-overlapping errors) to the winning trio. By the s2-vs-s23 rule (decorrelated-weak helps) this was the best remaining shot at >0.79459. Ready in `build_quad/`. |

**E50 campaign read-out (2026-07-15).** Final: **champtrio `t031+s17+s2` = 0.79459**, +0.0016 over the
previous champion single. Durable findings: **(1)** the 3.5k slice **mis-ranks** models for LB — it
picked s23 (#1 slice → 0.79130) over s17 (#5 slice → 0.79365) and over-predicted every regularizer
(MSD 0.7900→0.79121, ramp 0.7843→0.78552); never select members by it. **(2)** Ensembles beat every
single, and the lever is **decorrelation, not member strength** — s2 (0.78866 solo) *raises* the trio
while s23 (0.79130 solo) *lowers* it. **(3)** All DropHead-family regularizers (DropHead/layer-ramp/MSD)
lose to plain AWP on LB despite winning the 14k screen → family closed. **(4)** Seed variance on LB is
real (~±0.003 across identical-recipe seeds: 0.78866–0.79365).

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
