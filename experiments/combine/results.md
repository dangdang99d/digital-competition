# Branch: combine — maximize LB by stacking every proven win
Git branch: `research/combine` · Experiments: E8 (performance-max combo), E12 (weight averaging — the final stacking stage: E8-winning recipe × 3 seeds → soup → zip)
Goal: beat LB SOTA 0.77427 (submit_trainall_single, granite+alldata+filenames-only).

## Summary (at-a-glance)
Legend: 🏆 best · ✅ done · ⚠️ inconclusive. Matched CV = the untouched 3,500 held-out slice; champion-qwen3 on that slice = 0.7620 (uncal).

| Exp | Arm / recipe | Status | Result (uncal F1) | Δ | Verdict |
|---|---|:--:|---|---|---|
| E8a | granite-311m + richargs + full_data | ✅ | 0.7706 | +0.0086 vs champion-slice | ✅ WIN — beats champion & is the **fast** backbone (5:10) |
| E8b | qwen3-0.6B + richargs + full_data | ✅ | 0.7643 | +0.0023 vs champion-slice | ✅ weaker + near 9:06 budget cliff |
| E8a+LS | granite + richargs + full_data + **LS ε=0.1** | 🏆 | **0.7803** | +0.0097 vs E8a; **+0.0060 above LB SOTA on CV** | 🏆 **BEST of session — top submission** (zip `submit_0707_granite_ls.zip` built + verified) |
| E12 | model soup — 3 granite+LS seeds → average | ⚠️ | seeds 0.7803 / 0.7758 / 0.7628, **each on its OWN slice** | n/a | ⚠️ **BOTCHED** — `--seed` also reshuffles the split, so seeds aren't comparable and can't be soup-averaged cleanly. **Re-run with split held fixed** before any read |

## What the submission history proves (Notion export, 2026-07-07)
LB-verified gains, roughly independent axes:
- **Backbone:** qwen3-0.6B > bge-m3 > granite at equal recipe (+0.025 LB qwen3 vs bge @0703).
- **Serialization dir-removal:** bge v1→richmeta +0.018 LB, →richargs +0.021 LB.
  **richargs > richmeta on LB (0.76274 vs 0.76053) despite byte-identical CV** — the
  hidden test rewards arg-path stripping; CV can't see it. Ship richargs, not richmeta.
- **All-data training:** bge +75%-of-val → +0.014 LB; SOTA granite trained on ALL data,
  fold-CV≈LB (no gap). `--full_data` flag exists in src.finetune.
- **Depth pruning:** bge 24L→12L = +0.01 LB AND 1.85× faster. Compression can be free-or-better.
- **Budget cliff:** int8 qwen3 TIMED OUT; fp16 qwen3 = 9:06 of 10:00. Any addition to
  qwen3 inference cost is a submission-killer. richargs shortens inputs (helps);
  granite (5:10) and 12L-bge (2:29) have huge headroom.
- **Bias tuning:** correction — the SOTA zip DOES ship a `logit_bias.json` (fold-fit;
  e.g. read −0.15, glob +0.25). Its contribution is unmeasured. Our E8 runs stay
  uncalibrated per project decision; if E8 CV lands ≥ SOTA anyway, bias wasn't the driver.

## SOTA zip dissection (submissions/submit_names_single.zip)
- granite-311m = **ModernBERT, 22L / 768h**, fp32 safetensors 625MB, full tokenizer, @512.
- Serialization ("names", in-zip `render_sample`): `[META] tier pref turn budget-BUCKET
  elapsed-BUCKET lang ci git open=N loc openfiles=<basenames first 6> [HIST] ... [CUR]
  prompt` — ≈ our richmeta header. **History capped at the LAST 12 events**, lines
  `A[name] {compact-json args} -> result` — args keep FULL paths (arg-stripping =
  our richargs axis, UNTESTED on top of SOTA → E8's edge).
- Inference: plain batching (no length-sort), fp32+autocast, **fold-ensemble-ready**
  (averages every `model/granite-311m-v2-fold*`; this zip ships only fold1 → free
  headroom: multi-fold ensemble fits granite's 5-min budget easily).
- Trained fold-free on ALL data; CV proxied by separate fold runs (0.7725).

## E8 arms (both: tokenizer-prune, richargs serialization, --full_data, NO bias)
- **E8a granite-311m + richargs + full_data** — replicate SOTA then add our serialization
  axis. Cheap (~2h/3090), fast inference (huge budget headroom). Expected ≥0.775 LB.
- **E8b qwen3-0.6B + richargs + full_data @512** — best backbone + best serialization
  + max data. Expected ~0.78 LB. Watch inference budget (9:06 baseline; richargs shortens).
- Optional E8c (if E8b wins CV but time is tight): + depth-prune to buy budget back.

## Protocol notes
- `--full_data` evals on the untouched 25% of val (3.5k) — compare champion on the SAME
  3.5k slice (recompute from analysis/cache/qwen3_val_logits.npz), never cross-slice.
- LB is the final judge; CV deltas here are directional only (richargs lesson).
- Success = build zips for BOTH arms (recipe per EXPERIMENTS.md invariants, no logit_bias).

## Results
(append: date · arm · CV-slice score vs champion-slice · zip · LB when user submits)

**2026-07-07 · E8 RUNNING** — both arms full-FT @512, `--serialize richargs`, `--full_data`,
no calibration; eval on the untouched 25% (val=3,500). Baseline for harvest = champion qwen3
recomputed on the SAME 3,500 slice (from `analysis/cache/qwen3_val_logits.npz`), never cross-slice.
- **E8a granite-311m** — GPU 1, ~67% (epoch 2), ETA ~1h. `output/pat/ft_results_e8a.csv`.
- **E8b qwen3-0.6B** — GPU 0, ~33% (epoch 1), ETA ~3-4h (long pole). `output/pat/ft_results_e8b.csv`.
CV-slice scores + submission zips (recipe per EXPERIMENTS.md invariants, no logit_bias) on
completion; LB is the final judge (CV deltas directional only, per the richargs lesson).

### E8a Result — 2026-07-07 (granite-311m + richargs + full_data, HARVESTED)
Matched-slice harvest, uncalibrated raw-logit argmax, macro-F1 (sklearn `average="macro"`).
The 3,500 held-out ids are the untouched 25% that `--full_data` folds OUT of training:
`split_indices(y, seed=42)` → `va` (14,000); `train_test_split(va, test_size=0.25,
stratify=y_ids[va], random_state=42)` → `va_eval` (3,500). Champion qwen3 restricted to the
SAME 3,500 via `analysis/cache/qwen3_val_logits.npz` (positionally aligned to `va`).

- **E8a granite+richargs+full_data (uncal)** : **0.7706**  (`output/pat/ft_results_e8a.csv`; ignore cal 0.7789 per no-calibration policy)
- **champion-qwen3 on same 3,500 held-out** : **0.7620**  (n matched = 3,500 / 3,500 — all held-out ids present in cache, confirmed positionally AND by `id` string)
- **Δ (E8a − champion-slice)** : **+0.0086**
- directional CV-vs-LB (not matched): E8a 0.7706 vs LB SOTA 0.77427 = **−0.0037** (own held-out CV vs the LB's hidden test — informational only)
- figure: `experiments/combine/figures/e8a_vs_champion_slice.png`

**Verdict:** granite+richargs+full_data BEATS champion-qwen3 on the matched 3,500 slice by
+0.0086 macro-F1 — and granite is the FAST backbone (5:10 vs qwen3's 9:06, huge budget
headroom), so this is parity-plus at far lower inference cost: a clear win. E8b (qwen3 arm,
same recipe) still training (~2-3h remaining) and will be harvested separately.

### SUBMISSION BUILT — E8a+LS granite (submit_0707_granite_ls.zip) 2026-07-07
Packaged the session-best model (E8a+LS: richargs + full_data + label-smoothing ε=0.1).
- file: `submissions/submit_0707_granite_ls.zip` (887,082,660 B ≈ 846 MB, `unzip -t` OK)
- checkpoint: `.../granite-embedding-311m-multilingual-r2_e8a_ls_richargs_full/checkpoint-8314`
  (trainer `best_metric` = 0.78029)
- contents mirror the SOTA granite ref (`submit_names_single.zip`): `script.py`,
  `requirements.txt` (transformers==4.51.3 — the version that saved the ckpt; ref used 4.48.3),
  `serialize_variant.json` = `{"variant":"richargs"}`, `model/granite-311m-e8a-ls/` (config +
  fp32 model.safetensors + full tokenizer). NO logit_bias.json (no-calibration policy); raw argmax.
- VERIFICATION (all PASS): [1] serialization parity richargs (packaged serialize == `src.data`
  build_texts, byte-for-byte on test); [2] prediction parity (packaged `script.py` submission.csv
  == direct-model argmax, 5/5 test ids); [3] reproduced val macro-F1 on the exact `va_eval` 3,500
  = **0.7799** (≈ trainer 0.7803; fp32+autocast vs bf16-eval noise). Ready-to-upload (user decides).

### E8b Result — 2026-07-07 (qwen3-0.6B + richargs + full_data @512, HARVESTED)
Same matched 3,500 held-out slice, same protocol as E8a (uncalibrated raw-logit argmax,
sklearn macro-F1; the untouched 25% that `--full_data` folds OUT of training). Champion
qwen3 restricted to the SAME 3,500 via `analysis/cache/qwen3_val_logits.npz`.

- **E8b qwen3+richargs+full_data (uncal)** : **0.7643**  (`output/pat/ft_results_e8b.csv`; ignore cal 0.7773 per no-calibration policy)
- **champion-qwen3 on same 3,500 held-out** : **0.7620**  (identical matched-slice baseline as E8a)
- **Δ (E8b − champion-slice)** : **+0.0023**  (qwen3 richargs+full_data barely edges its own champion on CV)
- directional CV-vs-LB (not matched): E8b 0.7643 vs LB SOTA 0.77427 = **−0.0099** (own held-out CV vs the LB's hidden test — informational only)

### E8 SUMMARY — granite WINS the matched CV slice (2026-07-07)
Both arms on the SAME untouched 3,500 held-out slice, uncalibrated:

| arm | recipe | uncal macro-F1 | Δ vs champion-slice (0.7620) | inference |
|---|---|---|---|---|
| **E8a** | granite-311m + richargs + full_data | **0.7706** | **+0.0086** | 5:10 (fast) |
| **E8b** | qwen3-0.6B + richargs + full_data | **0.7643** | +0.0023 | 9:06 (near cliff) |

- **Headline:** on the matched CV slice, **E8a granite (0.7706) BEATS E8b qwen3 (0.7643) by +0.0063**,
  and it's the *fast* backbone (5:10 vs 9:06, huge budget headroom vs qwen3's 9:06/10:00 cliff).
  Both arms beat champion-qwen3-on-slice (0.7620); qwen3-richargs-full only by +0.0023.
- **⚠️ CV ≠ LB caveat:** the richargs lesson (richargs > richmeta on LB despite byte-identical CV)
  proves the hidden test rewards things CV can't see; and qwen3 carried a +0.025 LB edge over bge
  historically. So the LB granite-vs-qwen3 ranking may **invert** the CV ranking — CV deltas here
  are directional only. Both arms sit below LB SOTA on their own held-out CV (E8a −0.0037, E8b
  −0.0099 vs 0.77427), but that is own-CV vs hidden-test, not a matched comparison.
- **Recommendation:** build BOTH zips (recipe per EXPERIMENTS.md invariants, NO logit_bias) and
  let the user submit — granite is the CV winner AND the fast/cheap backbone (submit-safe budget),
  while qwen3 may still take the LB on its historical backbone edge. LB is the final judge.
- figure: `experiments/combine/figures/e8_combine_granite_vs_qwen3.png`
  (E8a 0.7706 vs E8b 0.7643 vs champion-slice 0.7620, dashed line at LB SOTA 0.77427)

### E8a+LS Result — 2026-07-07 (LS transfer to the full E8 recipe — TOP SUBMISSION CANDIDATE)
The E9 label-smoothing win (+0.0107 on the granite v1 screen) **TRANSFERS to the full E8
recipe.** Same matched 3,500 held-out slice, same protocol as E8a/E8b (uncalibrated
raw-logit argmax, sklearn macro-F1; the untouched 25% that `--full_data` folds OUT of
training). Recipe = granite-311m + richargs + `--full_data` + `--loss ls` (ε=0.1), NO bias.

- **E8a+LS granite+richargs+full_data+LS (uncal)** : **0.7803**  (`output/pat/ft_results_e8a_ls.csv`, tag `e8a_ls_richargs_full`; ignore cal 0.7856 per no-calibration policy)
- **E8a granite CE, same recipe (uncal)**          : **0.7706**  (baseline for the LS delta)
- **Δ LS transfer (E8a+LS − E8a)**                 : **+0.0097**  (label smoothing on top of the full recipe)
- vs **E8b qwen3 CE** 0.7643 → **+0.0160**;  vs **champion-qwen3-slice** 0.7620 → **+0.0183**
- directional CV-vs-LB (not matched): E8a+LS 0.7803 vs LB SOTA 0.77427 = **+0.0060 ABOVE SOTA on CV**
  — first arm of the session to clear SOTA on its own held-out slice (own-CV vs hidden-test, informational only)
- figure: `experiments/combine/figures/e8a_ls_transfer.png`
  (E8a+LS 0.7803 vs E8a 0.7706 vs E8b 0.7643 vs champion-slice 0.7620, dashed line at LB SOTA 0.77427, winner highlighted)

**Verdict:** **E8a+LS = 0.7803 is the best CV of the session** — LS stacks cleanly on the
3 confirmed axes (granite backbone × richargs × full_data), adds +0.0097 over the CE E8a,
and lands +0.0060 above LB SOTA on CV, all on the **FAST granite backbone** (5:10 vs qwen3's
9:06 — huge inference-budget headroom). **This is the top submission candidate.**
⚠️ **CV ≠ LB caveat still applies** (the richargs lesson: the hidden test rewards things CV
can't see, and CV deltas here are directional only) — but this stacks 3 LB-confirmed axes
plus the E9-confirmed LS win. Build the zip (recipe per EXPERIMENTS.md invariants, NO
logit_bias); LB is the final judge.
