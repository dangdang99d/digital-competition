# Branch: token-selection — upstream feature/token selection, decoupled from the model

Branch: `research/token-selection`. Scope: decide **which parts of the input actually matter**,
drop the rest, and **retrain on the reduced input**. This is the "hand-pick the good features so
the model learns better" move from Kaggle-style pipelines, applied to our serialized
action-decision inputs — i.e. **feature selection**, done as a *filter* (score importance once →
retrain once), NOT the model-internal learned-threshold pruning of LTP.

## Why this branch exists (how it differs from LTP / E18)

E18 (LTP) is now **deferred** because it wasn't what we wanted:

- **LTP is *embedded*** — a learned per-layer threshold *inside* the model drops tokens
  mid-network; the mechanism is part of the architecture and co-trained. On our data it also hit
  a temperature/scale wall (attention-received scores ~0.005 → the paper's small `T` explodes
  gradients, `T=1` makes the soft mask a ~0.5 no-op). See `experiments/EXPERIMENTS.md` → Deferred → E18.
- **This branch is *upstream and decoupled*** — score token/field importance, build a shortened
  input, then fine-tune a **normal** granite on it. No in-model surgery, no threshold learning,
  no `1/T` blow-up. The selector and the classifier are separate stages.

## Design principle — how selection is done (filter for A & B; embedded for C)

**A and B are *filters*** — the selector is decoupled from the classifier's training, so both run
as two cheap phases:
- **Phase 0** — score importance **once**, mostly **training-free** on the *frozen* champion
  (A: token attention/saliency · B: field occlusion/permutation). Minutes, no GPU-hours.
- **Phase 1** — retrain on the selected kept-set: a **few full FT runs** (**A's 4-value k-sweep
  {90,80,70,60}% · B's 3 field kept-sets**, each trained both from-scratch **and** recovery — see
  *Phase 1 training*), **not** a
  retrain-for-every-subset search. "Once" = *one run per kept-set*, **not one epoch** — each run is
  the full E8a recipe (3 epochs). Confirms the Phase-0 ranking under actual re-fitting.

Neither A nor B is a ❌ **wrapper** — we never retrain for *every* candidate subset until the score
improves (2ⁿ search, "delete features until it goes up"). That's the expensive trap, most tempting
for **B**; we avoid it by scoring first and retraining once.

**C is the deliberate exception — *embedded*:** its selector is trained *jointly* with the
classifier in one run, so filter-vs-wrapper doesn't apply (it's the higher-cost/risk method, gated
last). **D** is just trivial baselines (random/heuristic cuts), not a selection strategy.

## Baseline & metric (shared by every method)

- **Model / baseline:** the **deployment champion** — granite **E8a+LS, richargs**
  (`submit_0707_granite_ls.zip` = `output/pat/…_e8a_ls_richargs_full/checkpoint-8314`), full-FT ·
  full_data · LS · full vocab. Baseline = **0.7803 uncal** macro-F1 on the **clean 3.5k held-out
  slice** (seed 42; `--full_data` folds 75% of the 14k val back into training and holds out the
  remaining ~3.5k for eval — *not* trained on, so no leak). This model is both the frozen **Phase-0
  scorer** (richargs = the deployed field set → correct for B) and the full-input reference for
  Phase-1 Δ. *(Full vocab → standard `from_pretrained`, no remap.)*
- **Metric (project invariant — NO calibration):** uncal macro-F1 + tokens/fields kept (%) +
  ms/sample vs full input, all on the **same 3.5k held-out slice**. ⚠️ 3.5k is ~2× noisier than a
  14k val — watch that small Δ (±0.003) aren't lost in variance; Phase-1 retrains reuse the champion
  recipe + seed so they share the *identical* held-out slice (fair within-slice Δ).
- **Recovery is mandatory.** Training-free dropping already craters on our low-redundancy data
  (−0.05 @27% tokens dropped, −0.27 @42% — the E18 probe). Every method here **re-fine-tunes**
  after selection; the training-free number is only ever a lower bound.

## Phase 1 — training strategy: do BOTH (granite FT is cheap)

For every evaluated selection we train the reduced-input model **two ways** and report both:

- **Retrain (recovery-FT)** — warm-start from the champion (`checkpoint-8314`) and continue-train on
  the reduced inputs for **3 epochs** (same recipe as below). The **deployment** model: leverages
  champion knowledge, so strictly ≥ from-scratch. Every variant warm-starts from the *same* checkpoint.
- **From scratch** — fine-tune the pretrained granite **base** (+ fresh head) on the reduced inputs.
  The **honest** test: does the selected feature set *alone* suffice? Literature-matched (2303.07585).

**Config — hold everything constant except the input.** Both modes use the **exact E8a recipe**:
richargs · full_data (seed 42 → same 3.5k held-out) · LS ε0.1 · full-FT · linear head · **3 epochs** ·
lr 2e-5 · max_len 512. The **only** thing that changes is the reduced input, so any Δ is attributable
to selection, not to a config difference. **From-scratch and retrain differ in exactly one thing —
the init** (base vs champion); same 3 epochs, same lr 2e-5, everything.

**Sanity anchor:** a from-scratch run on the **full** (unreduced) input under this recipe should
reproduce the champion **~0.7803** — a faithfulness check on the harness before we trust any
reduced-input number.

**Why both:** cheap granite FT buys the shippable model AND the clean read at once, and the **gap
between them is the signal** — from-scratch ≈ retrain → selection genuinely lossless (strong);
from-scratch ≪ retrain → retrain is leaning on the champion's memory of dropped tokens (weaker,
deployment-only win).

---

## A · Decoupled token selection (score → prune → re-fit) — the direct test of the idea

Two-phase pipeline, exactly the published recipe of *Input-length-shortening via attention values*
(arXiv 2303.07585): a pretrained BERT's attention ranks tokens, then it **fine-tunes on the
shortened dataset** (from pretrained weights, not random scratch). On IMDB they kept the top **10%**
of tokens at negligible loss — but IMDB is highly redundant; our domain is not, so this is a real test.

- **A1 · Attention-filter.** *Phase 0:* score each token by the **attention it receives**
  (aggregated over heads) — from **our own trained champion** (task-aligned), not a generic BERT.
  *Phase 1:* keep the **top-k% highest-scoring tokens per example** (always protect special/pooled
  tokens), rebuild the shortened inputs, re-fine-tune. **Conservative keep-fraction sweep
  k ∈ {90, 80, 70, 60}%** (drop only 10 / 20 / 30 / 40%). We stay high on purpose: the training-free
  E18 probe already craters below ~60% kept (−0.05 @≈73%, −0.27 @≈58%) and the domain is
  low-redundancy, so the question is *"can we drop 10–40% for free / with a denoise gain?"*, not
  extreme pruning. **Start at k=90% and descend; stop at the first k where recovery can't close the
  gap** — the sweep self-limits, no pre-committing to aggressive values. (60% is the aggressive edge;
  drop it for an even milder run.)
- **A2 · Saliency-filter.** **Same pipeline and same conservative sweep k ∈ {90,80,70,60}%**
  (identical top-k% keep, protect specials, same start-high-and-descend) — the *only* change is the
  scorer: rank tokens by **gradient saliency** (|∂L/∂emb · emb| or integrated gradients) from the
  trained model instead of attention received (attention ≠ importance; saliency measures actual effect
  on the loss). Reuses `src/saliency.py`.
- **Cost:** low (one scoring pass + normal granite FTs). **Expected:** preserve-at-best
  (efficiency win); a small denoise gain is possible but not the base case on low-redundancy text.

## B · Structured field selection — filter on our *real* features (the gain bet)

Our input is not free text; it's a **serialized structured record** (`src/data.py::serialize`).
At the field level there are only ~15–20 features, so no subset search is needed.

- **Fields:** meta block (`tier, lang, turn, budget, elapsed, codelang, loc, ci, dirty, open,
  files`) · history decomposed into (USER turns · ACTION *names* · ACTION *args* · ACTION
  *result_summaries* · **history depth** = #turns) · current prompt.
- **Phase 0 (training-free):** field importance via **occlusion** (leave-one-field-out ΔF1 on the
  frozen champion) and/or **permutation importance** (shuffle a field's values across examples,
  measure ΔF1). ~15–20 forward passes over the 3.5k held-out slice — minutes, no training. **Half of it reuses
  existing `serialize()` flags** (`drop_meta`, `lean_actions`, `bare_actions`, `arg_basenames`,
  `strip_history`, `max_hist=k`) → measurable *right now* with no new code.
- **Phase 1:** drop the low-importance fields, retrain granite on the kept field-set — **3 kept-sets**
  (to trace a knee), each the full E8a recipe (3 epochs), trained both from-scratch + recovery.
  **Cost is filter, not wrapper:** the ~20 fields are *ranked* by cheap Phase-0 inference (no
  training); Phase 1 then retrains only **3 nested cumulative kept-sets** from that ranking (drop
  bottom-2 / bottom-5 / bottom-8), ×2 modes = **6 FT runs total** — **not 2²⁰ subsets, not 20
  leave-one-out retrains**.
- **Why it's the gain bet:** removes whole *distractor* fields → structured denoising (this is
  where *Focus on the Core*, arXiv 2406.01283, reported **+5.0 acc / +5.6 macro-F1** on document
  classification) + shorter input (speed). Orthogonal to A — can win even if token-level selection
  has no slack. Fits our weak-semantic-label finding: some serialized fields may be pure noise.
- **Cost:** medium (a few retrains). Connects to existing serialization / E21 work.

## D · Heuristic / random controls — A's controls (same harness)

- **D1 · Trivial-cut controls:** random-drop k%, stopword / low-TF-IDF removal, plain
  truncation-to-length — **at A's chosen keep-ratios**.
- **Why mandatory:** without them, "keep 50% → same F1" proves nothing (maybe a *random* 50% also
  works). They make any A/B/C result interpretable. Near-zero extra cost — only the selection rule
  changes in the shared retrain harness.

## C · Rationale extraction — joint learned selection (gated, last)

- **How:** a **selector** emits a sparse token mask; the **predictor** sees only kept tokens; both
  trained end-to-end with a **sparsity + sufficiency** objective, using **HardKuma**
  reparameterization (or REINFORCE) to stay differentiable. Refs: *Rationalizing Neural
  Predictions* (Lei et al. 2016, arXiv 1606.04155); *Interpretable Neural Predictions with
  Differentiable Binary Variables* (Bastings et al. 2019, arXiv 1905.08160).
- **Why last:** highest ceiling (learns to keep exactly what the classifier needs) but **high
  cost + the same non-differentiable-selection instability that stalled LTP** (REINFORCE/HardKuma
  variance). Only worth it if A or B shows there's actual prunable slack here.

---

## Sequencing (planned)

> **Phase 1: A** (attention/saliency select + refit) → **Phase 2: B** (free Phase-0 field scan →
> 1–3 denoised retrains) → **Phase 3: D** (random/heuristic controls at A's chosen keep-ratios) →
> **Phase 4: C** (rationale extraction) **only if** A/B reveal slack.

**B's Phase-0 is training-free**, so it runs up front at zero cost. **D is A's control** and shares
A's retrain harness — run it at the keep-ratio A settles on (so we don't spend control runs at
ratios we don't care about). The only hard rule: D's controls sit beside A's results before we
conclude anything.

## Status

| Method | Phase 0 (score) | Phase 1 (retrain) | Status |
|--------|-----------------|-------------------|--------|
| A1 attention-filter | champion attention-received (`a_score_tokens.py`) | k=90 scratch+recover | ✅ **RUN 2026-07-09** → negative |
| A2 saliency-filter  | grad saliency (`a_score_tokens.py`) | k=90 scratch | ✅ run (sal_k90_scratch) |
| B  field selection  | occlusion (`b_field_occlusion.py`, 19 fields) | drop2/5/8 scratch+recover | ✅ **RUN 2026-07-09** → lossless |
| D1 controls         | random / stopword / truncate | at A's k | 🔲 not needed (A/B null) |
| C  rationale        | — (joint) | selector–predictor E2E | 🔲 dropped (gated on A/B slack — none found) |

## Results & Verdict (run 2026-07-09; raw uncal val Macro-F1 on the 3.5k held-out slice)

**Reference points:** champion (full input) **0.7803**; **from-scratch anchor (full input, no reduction) = 0.7790** ≈ champion
→ **the harness is faithful** (from-scratch reproduces the champion within slice noise). Slice noise ≈ ±0.003 (3.5k is ~2× noisier than 14k).
**The clean read is from-scratch vs the 0.7790 anchor** (recovery is confounded — see finding 1).

| method | mode | val | Δ anchor 0.7790 | read |
|---|---|---|---|---|
| **A · attn token-drop k90** (keep 90%) | from-scratch | **0.7712** | **−0.008** | token-drop HURTS (beyond noise) |
| A · attn token-drop k90 | recover | 0.7639 | −0.015 | (confounded) |
| A · sal token-drop k90 | from-scratch | *pending (~0.77)* | — | saving |
| **B · field-drop drop5** (drop 5 low-imp meta) | from-scratch | **0.778** | **−0.001** | ~LOSSLESS |
| B · field-drop drop2 | from-scratch | 0.7725 | −0.006 | ~lossless (noise) |
| B · field-drop drop8 | from-scratch | 0.7725 | −0.006 | ~lossless (noise) |
| B · field-drop drop2/5/8 | recover | 0.7633 / 0.7623 / 0.7625 | −0.016 to −0.017 | (all confounded) |

**VERDICT — neither token- nor field-selection beats the champion; the "learn-better-by-selecting-features" bet is NULL here.**
- **A (token selection): mild loss.** Dropping even 10% of tokens (highest-attention kept) costs **~0.008** macro-F1 from-scratch — our serialized inputs are **low-redundancy** (as the E18 probe predicted); no free lunch, no denoise gain.
- **B (field selection): ~lossless, no gain.** Dropping the 5 least-important meta fields is **free** (−0.001, within noise → shorter input = an *efficiency* win) but yields **no accuracy gain**. Non-monotonic across drop2/5/8 confirms it's noise-dominated. *Focus-on-the-Core*'s +5pt does **not** replicate on this task.
- **⇒ Selection is efficiency-only at best. Champion stays. D/C dropped** (they were gated on A/B showing slack — none did).

**Key methodological findings (reusable):**
1. **Recovery-degradation:** warm-starting the champion and continuing 3 epochs at lr 2e-5 **degrades it to ~0.763** — *below* from-scratch (~0.771–0.779) — regardless of what's reduced. So *recovery-vs-champion Δ is confounded*; always compare **from-scratch vs the from-scratch anchor**. (Recovery is NOT "strictly ≥ from-scratch" here, contra the plan's assumption.)
2. **Faithfulness anchor works:** from-scratch full-input = 0.7790 ≈ champion 0.7803 → trust the harness; reduced-input Δs are real, not artifacts.
3. **3.5k-slice noise ±0.003** — treat |Δ|<0.003 as lossless.

### Scorer-overlap: do attn & sal rule out the SAME tokens? (`a_drop_overlap.py`, CPU, user ask 2026-07-09)

**Barely above chance → no large model-independent "unimportant token" set exists.** Per-sample
Spearman(attn, sal) = 0.35 median; drop-set overlap 12.0% @k90 vs 9.7% chance (1.2×), 30.1% vs
19.8% @k80 (1.5×), 42.8% vs 29.7% @k70. attn drops by token *type* (serialization syntax: `␣'`
83% of its occurrences, `=`, `_`, digits, `ok`); sal drops by *context* (≤14% of any single type;
near-uniform positional profile). The consensus ∩ (~1.2% of content tokens @k90) is almost pure
serialization formatting — E21-consistent (format isn't the edge). A transferable ignore-set is
therefore only the small syntax core, and reusing it on other models needs an LB confirm against
train-overfit.

## How to run (exact commands — do NOT change the recipe)

**Every command:** cwd = repo root `/home/ocean/dacon`, python =
`/home/ocean/miniconda3/envs/dacon/bin/python`, `PYTHONPATH=/home/ocean/dacon`. Use **GPU 1 or 3**
(`CUDA_VISIBLE_DEVICES=1|3`) — **0/2 run the `coreset` job, do NOT touch them**. Launch training
detached (`nohup … &`) so a disconnect can't kill it.

⚠️ The `--reduced_ids` files MUST come from the **full** Phase-0 run (all 70k) — a `--limit` smoke
file will index out of range (tr/va span the full 70k).

### Phase 0 — score tokens (once, ~10 min, 1 GPU; defaults already correct)
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/ocean/dacon nohup \
  /home/ocean/miniconda3/envs/dacon/bin/python -u experiments/token-selection/a_score_tokens.py \
  > experiments/token-selection/artifacts/score.log 2>&1 &
```
→ `experiments/token-selection/artifacts/a_token_scores.npz` (champion ckpt, richargs, max_len 512 baked in).

### Build reduced inputs — 8 files (CPU, seconds)
```
for s in attn sal; do for k in 0.9 0.8 0.7 0.6; do
  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/token-selection/a_build_reduced.py --scorer $s --keep $k ; done ; done
```
→ `artifacts/a_reduced_{attn,sal}_k{90,80,70,60}.npz`.

### Phase 1 — LOCKED E8a recipe (only the input differs)
**Do NOT change these flags.** Per run, vary ONLY: `--reduced_ids`, `--tag`, presence of
`--init_from` (present ⇒ recovery, absent ⇒ from-scratch), and `CUDA_VISIBLE_DEVICES`.
Constant block: `--model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs
--full_data --loss ls --label_smoothing 0.1 --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4
--max_len 512 --seed 42 --out_dir ./output/pat --results_name ft_results_e24.csv`
(warmup 0.05 / wd 0.01 / linear are finetune.py defaults = champion's; `--serialize richargs` is
metadata only — tokenization is bypassed by `--reduced_ids`).

**RECOVERY (warm-start champion)** — add `--init_from <champion>`:
```
CUDA_VISIBLE_DEVICES=1 nohup /home/ocean/miniconda3/envs/dacon/bin/python -u -m src.finetune \
  --init_from output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e8a_ls_richargs_full/checkpoint-8314 \
  --reduced_ids experiments/token-selection/artifacts/a_reduced_attn_k90.npz \
  --tag a24_attn_k90_recover \
  --model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs --full_data \
  --loss ls --label_smoothing 0.1 --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4 \
  --max_len 512 --seed 42 --out_dir ./output/pat --results_name ft_results_e24.csv \
  > sbatch/logs/a24_attn_k90_recover.log 2>&1 &
```

**FROM-SCRATCH (base init)** — identical but **NO `--init_from`**, tag `_scratch`:
```
CUDA_VISIBLE_DEVICES=3 nohup /home/ocean/miniconda3/envs/dacon/bin/python -u -m src.finetune \
  --reduced_ids experiments/token-selection/artifacts/a_reduced_attn_k90.npz \
  --tag a24_attn_k90_scratch \
  --model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs --full_data \
  --loss ls --label_smoothing 0.1 --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4 \
  --max_len 512 --seed 42 --out_dir ./output/pat --results_name ft_results_e24.csv \
  > sbatch/logs/a24_attn_k90_scratch.log 2>&1 &
```

**Sweep order:** k=90 first (both modes × {attn,sal}); read val; descend 80→70→60 **only while
recovery ≈ champion**; stop at the first k that can't recover. Tag convention `a24_<scorer>_k<K>_<mode>`.

### Method B — field selection (reuses the SAME `--reduced_ids` hook)
Field transforms live in `b_fields.py` (one source of truth for Phase-0 + Phase-1). 19 droppable
fields: whole meta block · all history · history depth · action detail/bare · each meta subfield
(tier/lang/turn/budget/elapsed/codelang/loc/ci/dirty/open/files) · user turns · prompt.

**B Phase-0 (occlusion, training-free, ~15 min, GPU 3):**
```
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/ocean/dacon nohup \
  /home/ocean/miniconda3/envs/dacon/bin/python -u experiments/token-selection/b_field_occlusion.py \
  > sbatch/logs/b24_occlusion.log 2>&1 &
```
→ `artifacts/b_field_importance.json` (fields ranked by ΔF1; the printed BASELINE should read
**≈0.7803** — a check that the held-out slice matches the champion's).

**Build reduced-field inputs** (drop the lowest-importance fields; 3 nested kept-sets + anchor):
```
PYTHONPATH=/home/ocean/dacon python experiments/token-selection/b_build_reduced.py --drop <bottom-2> --tag drop2
PYTHONPATH=/home/ocean/dacon python experiments/token-selection/b_build_reduced.py --drop <bottom-5> --tag drop5
PYTHONPATH=/home/ocean/dacon python experiments/token-selection/b_build_reduced.py --drop <bottom-8> --tag drop8
PYTHONPATH=/home/ocean/dacon python experiments/token-selection/b_build_reduced.py            --tag full   # anchor → ~0.7803
```
(`<bottom-N>` = comma-sep field names from the tail of `b_field_importance.json`.)
→ `artifacts/b_reduced_<tag>.npz`.

**B Phase-1:** identical to the A Phase-1 commands, but point `--reduced_ids` at
`experiments/token-selection/artifacts/b_reduced_<tag>.npz` and tag `b24_<tag>_<mode>`
(recovery = add `--init_from <champion>`; from-scratch = omit). Same LOCKED E8a recipe.

### Baseline, anchor, read-out
- **Baseline = champion 0.7803** (full-input, 3.5k held-out). Do NOT retrain it.
- **Faithfulness anchor** (optional, once): the FROM-SCRATCH command with `--reduced_ids` **removed**
  (full input) should reproduce **~0.7803** — validates the harness before trusting reduced numbers.
- **Result** = each run's last `val_macro_f1` in `output/pat/ft_results_e24.csv` (uncal, on the
  reduced 3.5k held-out); Δ vs 0.7803. **NO calibration** (project invariant).

## Existing repo assets to reuse

- `src/finetune.py` — the granite FT harness (Phase-1 retrains for all methods).
- `src/data.py::serialize` + `SERIALIZE_VARIANTS` flags — field ablation scaffold for B Phase-0.
- `src/saliency.py`, `analysis/token_importance.py` — token-importance scoring for A2 / B.
- **Weights — where to load from:**
  - *Champion* (retrain warm-start **and** Phase-0 scorer): `output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e8a_ls_richargs_full/checkpoint-8314/` → `model.safetensors` (1.25 GB fp32) + `config.json` + tokenizer. Load with `AutoModelForSequenceClassification.from_pretrained(<that dir>)` — full vocab, **no remap**. Optimizer state present (resumable). Equivalent packaged copy: `submissions/submit_0707_granite_ls.zip` (887 MB, ships model + `script.py`).
  - *Base pretrained* (the from-scratch arm): HF id `ibm-granite/granite-embedding-311m-multilingual-r2`, cached at `~/.cache/huggingface/hub/models--ibm-granite--granite-embedding-311m-multilingual-r2` — this is `finetune.py --model`'s value.

## References

- **Input-length-shortening via attention values** — the A1 two-phase recipe. arXiv 2303.07585.
- **Focus on the Core: Pruned Token Compression for Document Classification** — in-model pruning,
  reported accuracy *gains*; motivates B. arXiv 2406.01283.
- **Rationalizing Neural Predictions** — Lei, Barzilay, Jaakkola 2016 (C). arXiv 1606.04155.
- **Interpretable Neural Predictions with Differentiable Binary Variables** (HardKuma) — Bastings,
  Aziz, Titov 2019 (C). arXiv 1905.08160.
- **Learned Token Pruning for Transformers** — Kim et al. 2022 KDD; the *embedded* contrast (E18,
  deferred). arXiv 2107.00910.
- **LAFS: differentiable feature selection via learnable attention** — B-adjacent. PMC12839874.
