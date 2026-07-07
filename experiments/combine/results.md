# Branch: combine — maximize LB by stacking every proven win
Git branch: `research/combine` · Experiments: E8 (performance-max combo), E12 (weight averaging — the final stacking stage: E8-winning recipe × 3 seeds → soup → zip)
Goal: beat LB SOTA 0.77427 (submit_trainall_single, granite+alldata+filenames-only).

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
