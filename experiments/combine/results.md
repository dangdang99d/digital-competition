# Branch: combine — maximize LB by stacking every proven win
Git branch: `research/combine` · Experiment: E8 (performance-max combo)
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
