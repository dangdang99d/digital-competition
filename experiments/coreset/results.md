# Group: coreset — drop-noisy data selection to raise macro-F1

Group `research/coreset` (logical only — **no git branch, no commit**, per the shared-tree
protocol). Objective: **increase performance (macro-F1)** by training on a cleaned subset, not
reduce compute. This is the *drop-noisy / data-centric denoising* family (remove mislabeled
samples), **not** classic keep-hard coreset. Screen backbone: granite-311m (fast, E9/E11
precedent); confirm the winner on qwen3. Baseline: granite CE full-56k control = **0.7458**.

---

## §1 · Suitability confirmation — DOES drop-noisy fit our objective?

**Verdict: 🟡 CONDITIONAL GO — only in the *surgical* form; the blanket form is predicted to HURT.**

Drop-noisy raises macro-F1 **iff** the removed samples are genuinely *mislabeled* (label ≠ true
class). It **hurts** when errors come from *intrinsic ambiguity* (input maps to several valid
labels) — the test set carries the same ambiguity, and dropping ambiguous points removes real
signal (cartography: ambiguous examples drive generalization). So the question is: how much of our
error is removable noise vs irreducible ambiguity?

**Probe (no training; on the qwen3 champion's held-out 14k val, + hist0 as an independent 2nd
model).** Figure: [suitability_error_profile](figures/suitability_error_profile.png).

| Signal | Value | Reading |
|---|---|---|
| val error rate | 23.5% (3285/14000) | the room, if any |
| errors that are **low-confidence** (MSP<0.5) | **40.5% of errors** | near-boundary **ambiguity → irreducible** |
| **confident-wrong** (MSP>0.9) | 2.99% of val | model sure but label disagrees → mislabel candidate |
| **2-model consensus ≠ label** (qwen3 & hist0 agree on a non-label class) | 16.2% of val | shared-error *or* mislabel |
| …**AND both confident** (>0.6) | **6.26% of val** | **removable-noise upper bound** |
| errors concentrate on | list_directory 48% · read_file 46% · glob_pattern 43% · grep_search 32% | **synonymous file-ops → intrinsic confusability** |

**Interpretation:**
1. **Better odds than pure ambiguity would predict.** Two *independent architectures* (qwen3
   decoder, bge-m3 encoder) confidently agree on a non-label class for **~6%** of val — the
   classic confident-learning label-error signature. So there *is* a removable-noise population,
   more than the ~2% I feared. This is the real argument FOR trying it.
2. **But the dominant error mode is intrinsic confusability**, not corruption: errors pile on
   `read_file / list_directory / glob_pattern / grep_search` — near-synonymous "inspect the
   codebase" actions where several labels are all reasonable and the sim picked one. Dropping
   these can't help (test has the same confusability) and 40% of errors are low-confidence.
3. **Macro-F1 risk is the killer for the naive version.** The errors concentrate on classes that
   are already hard/smaller. The user's original phrasing — "train only on samples the model gets
   right" = drop **all 23.5%** — would gut those classes' training signal → **predicted to LOWER
   macro-F1**, which weights every class equally.

**Consequence for the method (this shapes §2):**
- ✅ **Suitable:** *surgical* denoising — remove/relabel only the **~6% confident-consensus
  mislabels** (Group 1: confident-learning), and cartography variants that **keep ambiguous +
  easy, drop only confident-hard/mislabeled**. Realistic ceiling is modest (~6% of labels).
- ❌ **Not suitable:** blanket "drop everything the model predicts wrong" — drops the ambiguous
  boundary + rare-class signal; expected macro-F1 **loss**.
- **First test is decisive & cheap:** cleanlab confident-learning to *quantify* the true mislabel
  rate + one surgical retrain. If cleaning the ~6% gives ≥+0.003, build out; if flat, the
  direction is confirmed dead and we stop before the Group-3 engineering.

---

## §2 · Method plan + compute (granite screen; base retrain ≈ 1–1.5 GPU-h, calibrate on run 1)

Shared foundation: 1 instrumented baseline run (train full-56k + log per-epoch dynamics) ≈ 1.5
GPU-h — unlocks cartography/forgetting/AUM/EL2N scoring for free. Needs `finetune.py --keep_indices`
+ a dynamics callback (both TBD).

| # | Group | Method | **Scoring model** (selects keep-set) | **Retrain model** (trained on keep-set) | Compute (GPU-h) | Eng | Priority |
|---|---|---|---|---|---|---|---|
| C1 | 1 | **Confident Learning (cleanlab)** — targets the ~6% directly | **granite** k-fold OOF (4× fresh) | fresh granite | ~5 (4-fold OOF + 1 retrain) | low | 🥇 **HIGH — decisive** |
| C1b | 1 | **Reference / cross-model consensus** — reference model confidently disagrees with the label | **qwen3** (pruned champion, cached) for the val ruler · +hist0 optional consensus · **qwen3 k-fold** for train-select | fresh granite | val ruler ~0 (cached); train-select = qwen3 k-fold (~12) | low | med — §1 estimator; **also the eval ruler** |
| C2 | 1 | AUM mislabel ranking | **granite** dynamics (baseline run) | fresh granite | ~1 (+shared) | low | med |
| C3 | 1 | CHE-refined selection | **granite** dynamics | fresh granite | ~1 | med | low |
| C4 | 2 | Dataset Cartography (keep easy+amb, drop conf-hard) | **granite** dynamics (baseline run) | fresh granite | ~3–4 (3 retrains) | low | 🥈 high |
| C5 | 2 | Forgetting events | **granite** dynamics (baseline run) | fresh granite | ~1 | low | med |
| C6 | 2 | EL2N / GraNd | **granite** dynamics (early epoch) | fresh granite | ~2 | low | med |
| C7 | 2 | PVI | **granite** OOF (reuse C1) | fresh granite | ~1.5 | med | low |
| C8 | 3 | Co-teaching | integrated — 2× fresh **granite** (co-train) | = the co-train run | ~2 | HIGH | low (gated) |
| C9 | 3 | MentorNet | integrated — mentor + student **granite** | = the run | ~3 | HIGH | low (gated) |
| C10 | 3 | DivideMix | integrated — 2× fresh **granite** | = the run | ~3 | VERY HIGH | ⛔ gated |
| C11 | 3 | SELFIE | integrated — fresh **granite** | = the run | ~1.5 | HIGH | low (gated) |

**Model columns:** *Scoring model* = what produces the keep-set — **all granite** (fresh, from HF
base) **except C1b's reference = qwen3 (pruned champion)**, cached on val (a qwen3 k-fold if it
selects train samples); **hist0 is only an optional consensus add-on, not the default** (weak,
0.7504). *Retrain model* = **fresh HF-base granite** for the whole screen; the **winning recipe is
then confirmed on qwen3**. Group 3 unify scoring+retrain in one training loop. Eval-side val hardness
ruler = **qwen3** (pruned champion; +hist0 optional consensus) — except Group-1 method-consistent
(cleanlab/PVI on val).

**Weight locations (pre-empting the "where are the weights" problem — they're in `submissions/*.zip`,
NEVER `output/`; the `output/` champion dirs hold only config+tokenizer, no `model.safetensors`):**
The coreset plan needs **NO fine-tuned weight loading** — every granite scoring/retrain trains
**fresh from HF base** (`ibm-granite/granite-embedding-311m-multilingual-r2`, HF cache), the qwen3
winner-confirm trains **fresh from HF base** (`Qwen/Qwen3-Embedding-0.6B`), and the C1b reference uses
**cached logits**, not weights:
- qwen3 reference (val ruler) → `analysis/cache/qwen3_val_logits.npz`
- hist0 reference (**optional** consensus add-on only) → `analysis/cache/hist0_val_logits.npz`

IF weights are ever actually needed (only for re-extraction — a fresh qwen3 k-fold still trains from HF base):
- **qwen3 champion (0.7682, standard-split — the val-OOF model matching the cache):**
  `submissions/submit_0703_qwen3_pruned.zip` → `model/qwen3-0.6b/` (pruned embedding + `remap.npy` +
  full HF tokenizer — load recipe in memory `qwen3-champion-weights-loading`).
- **hist0 (bge-m3) val-OOF:** ⚠️ **only the cache is safe.** `submissions/submit_0705_bgem3_full.zip`
  (`model/bgem3-hist0/`) is the **full-data** variant (it SAW the 14k val) → it is NOT the val-OOF
  model behind the cache; do not reload it for val scoring. The standard-split hist0 weights live only
  in `output/` (possibly unsynced) — so for hist0, use the cached logits, full stop.

**Total ≈ 25 GPU-h** (parallel over GPUs 1/2/3 ≈ 8–9 h wall-clock) + ~3 GPU-h qwen3 confirm.

**Execution order:** foundation → **C1 (cleanlab) + C4 (cartography)** first (they *are* the
surgical test) → if either clears +0.003, expand Group 2 → Group 3 loops **only if a denoise
gain is proven** (don't build DivideMix on spec). Every retrain reports **macro-F1 + per-class F1
on the rare/confusable classes** (the macro-F1 risk from §1).

## §3 · Decisions — training/scoring models + evaluation methodology (2026-07-08 discussion)

**Training & scoring models — always fresh from HF base granite, NEVER a fine-tuned checkpoint:**
- **Retrain every keep-set from HF base** `ibm-granite/granite-embedding-311m-multilingual-r2`,
  v1, standard split, CE — identical init to the baseline. *Why:* apples-to-apples (only the
  training data may differ); a full-data fine-tuned granite has **already absorbed the noise**, so
  continuing on a cleaned subset can't un-see it — drop-noisy only works if the model **never fits
  the noise**.
- **Score with freshly-trained granite, NOT the existing fine-tuned granites** (E8a / granite_ls /
  TAPT…): they trained on these 56k → their train predictions are **in-sample / memorized** →
  invalid for confident learning (a memorized model calls every label "correct"); the `full_data`
  ones even saw val. So **Group 1 OOF = k-fold fresh granite**; **Group 2 dynamics = the one
  instrumented baseline run**. Scoring model must also match the retrain recipe (fresh granite, v1, CE).
- **Cost reconciliation** (~1 GPU-h/fresh granite run, smoke-confirmed): Group-2 *scoring* ≈ free
  (shares the baseline run). **cleanlab k-fold ≈ 4 GPU-h is the one expensive scorer.** The
  **retrains (~1 GPU-h each) are the real GPU-hour cost**, not the scoring. → stage **cheap Group-2
  first**; pay for cleanlab only if Group-2 shows a gain.

**Evaluation — two-measurement design (user request):**
- **Primary = overall macro-F1 + per-class F1** (rare-class guard). This is the **decision metric**
  and the **literature standard** — cartography / cleanlab / data-pruning report *overall* test
  performance, **not** a test easy/hard split.
- **Mechanics:** one forward pass over all 14k val → per-sample predictions. macro-F1 is then
  computed over **subsets via boolean masks** (all / easy / hard). The partition changes *only which
  indices are aggregated*, never the inference. Report **n and % in each tier**.
- **The easy/hard val split is OUR extension** (for goals 2 & 3) — not part of the selection methods.
- **Val hardness ruler — what's computable where:**
  - **Group 2** (cartography / AUM / forgetting / EL2N) = **training dynamics → NOT computable on
    val** (a held-out sample has no training dynamics).
  - **Group 1** (cleanlab / PVI) = **OOF → IS computable on val** (the baseline held val out, so its
    val probs are out-of-sample). Evaluate Group-1 keep-sets **method-consistently**:
    `cleanlab.find_label_issues(val_labels, val_OOF_probs)` / PVI-on-val give the method's own
    easy/hard on val (≈ free — reuse the baseline val logits).
  - **NEVER use the drop-noisy model's own MSP for the partition — circular** (that model is trained
    to be confident on easy). A "reference model margin" is just the OOF-confidence signal = a
    **Group-1-family** method, not a neutral ruler and not Group-2.

**New method surfaced — C1b · Reference / cross-model consensus.** The "reference model" we kept
returning to is itself a distinct, grounded selection method: flag a sample as mislabeled when
independent reference model(s) that **held it out** confidently predict a class ≠ its label
(model-consensus / committee dataset cleaning, e.g. ImageNet-cleanup-by-consensus). It's
**Group-1-family** (OOF-confidence) but distinct from **C1 cleanlab** (same-backbone k-fold
*self*-confidence) in using *external / cross-model* agreement — more robust to one model's bias, and
can use a *stronger* reference. **It's exactly what §1 used** (qwen3 + hist0 → the ~6% estimate).
**Default reference = qwen3 (pruned champion) alone** — the strongest val-OOF model, cached logits,
and cleanlab-on-qwen3 is a principled single-model method; **hist0 is an optional consensus vote for
extra precision, NOT the default** (weak at 0.7504 → modest marginal value). Note `submit_0703_qwen3_pruned`
is **vocab-pruned only** = behaviorally the champion. **Dual role:** (a) a **selection** method — to score *training* samples it needs out-of-sample
reference preds (k-fold or a differently-trained/stronger backbone); (b) the **eval-side val hardness
ruler** — the baseline held val out, so on val it's free.
- **Two distinct "hard" notions (they behave differently at eval):**
  1. **Mislabeled** (Group 1): hard-val labels are *suspect* → **F1 on hard-val is confounded** (a
     better model that predicts the true class scores *worse* against the wrong label). ⇒ the
     **meaningful number is F1 on the clean/easy subset**; report the hard *count / %* but don't
     over-read its F1. A specialist **cannot** rescue a mislabeled sample.
  2. **Difficult-but-correctly-labeled** (ambiguity/confidence): labels are fine → F1-on-hard is
     meaningful, and this is the right axis for **goal 3** (a complementary specialist *can* help).

**The three goals (user's framing):**
1. **How dropping hard *training* samples improved performance** → read on the **clean/easy** subset + overall.
2. **What the cleaned model sacrifices on hard samples** → its hard-tier performance (difficulty notion;
   the mislabel-tier F1 is confounded, per above).
3. **Detect hard at inference → route to a different, adequate model** (cascade). **SEPARATE
   experiment.** Unlike E6 (two *architectures*, correlated errors), the easy-model and hard-model are
   trained on **complementary data** → their errors may genuinely decorrelate. Gated on the primary
   result; held to the **au + first-step** robustness bar (E6 lesson).

**Per keep-set, report:** overall macro-F1 · per-class F1 · clean/easy-tier F1 (+n, %) · hard-tier F1
(+n, %, caveated) · that method's train drop count & %.

## Implementation status (Group 1 + 2 — 2026-07-08)

Plumbing added to `finetune.py` (opt-in flags, non-invasive; smoke-tested ✓):
- `--keep_indices <npy>` — restrict training to tr ∩ indices (drop-noisy retrain).
- `--log_dynamics <npz>` — per-epoch train softmax callback (source for cartography/AUM/forgetting/EL2N).

Scorers in `experiments/coreset/` (all syntax-clean; base granite run ≈ 1 GPU-h, confirmed):
| Script | Methods | Needs | Cost |
|---|---|---|---|
| `confident_learning.py` | **C1 cleanlab** | k-fold OOF (4 granite trainings) | ~5 GPU-h |
| `score_dynamics.py` | **C2 AUM · C4 cartography · C5 forgetting · C6 EL2N** | 1 instrumented baseline run's `--log_dynamics` npz | ~1 GPU-h + retrains |
| `pvi.py` | **C7 PVI** | reuses C1's OOF (null model ≈ class prior) | free |
| C3 CHE | — | deferred (refinement on top of C2) | — |

**Run plan:** (1) instrumented baseline granite run (full 56k, `--log_dynamics`) → also the drop-noisy baseline. (2) `score_dynamics` + `pvi` → keep-sets. (3) `confident_learning` (k-fold, parallel). (4) retrain granite per keep-set (`--keep_indices`) → macro-F1 + per-class F1 (watch rare classes) vs baseline. Decisive first read: C1 (cleanlab) + C4 (cartography).

## Run (orchestrator — for the /loop)

**🟢 LAUNCHED 2026-07-08 ~22:05, in progress** (via /loop, 10-min babysit). All 4 GPUs busy:
base(+dynamics) on GPU0, cleanlab fold0/1/2 on GPU1/2/3, fold3 queued. ~4 h wall-clock ETA.
`coreset_eval.py` **written + verified** (compiles; qwen3 ruler val-acc 0.7653 sanity ✓; tiers
easy 10714 / hard 3286 [suspect 1386 · ambig 1900] / cleanlab-clean 11744) — runs after GPUs free.
Prior verify: plumbing (`--keep_indices`/`--log_dynamics`), scorers, transformers/cleanlab all pass.

- **Orchestrator:** `experiments/coreset/run_coreset.py` — GPU-slot queue keeping all 4 GPUs busy
  over the DAG: baseline(+dynamics) + 4 cleanlab folds → CPU scorers inline (score_dynamics /
  cleanlab-merge / pvi) → **10 keep-set retrains** (cl · cart06/15 · aum06/15 · el2n06/15 · forget
  · pvi06/15). ~15 granite runs ≈ ~4 h wall-clock on 4 GPUs.
- **Launch:** `cd /home/ocean/dacon && nohup /home/ocean/miniconda3/envs/dacon/bin/python \
  experiments/coreset/run_coreset.py > sbatch/logs/coreset_orch.out 2>&1 &`
- **Monitor:** `tail sbatch/logs/coreset_orch.out` (launches/completions/final macro-F1 harvest);
  per-job logs `sbatch/logs/coreset_<name>.out`.
- **Loop's job:** (1) launch the orchestrator if not running; (2) monitor + report; (3) **write
  `coreset_eval.py`** (per-class F1 + easy/hard tiers via qwen3 ruler / cleanlab-on-val, §3) while
  training runs; (4) when retrains finish, harvest raw `best val Macro-F1` vs baseline **0.7458** +
  per-class → fill this doc. **Use the RAW macro-F1; ignore finetune's calibration output** (§3).
- **Gotchas:** `confident_learning.py` needs `PYTHONPATH=/home/ocean/dacon`; score_dynamics/pvi
  don't. finetune retrains write `output/pat/ft_..._coreset_rt_<name>/`.

## Summary
| Exp | Test | Status | Result | Verdict |
|---|---|:--:|---|---|
| §1 | suitability probe (removable noise vs ambiguity) | ✅ | ~6% consensus mislabels; 40% errors low-conf; errors on synonymous file-ops | 🟡 CONDITIONAL GO — surgical only |
| §3 | decisions: fresh-granite scoring/retrain · eval (overall F1 primary + Group-1 method-consistent easy/hard) | ✅ | recorded | reference |
| C1,C2,C4,C5,C6,C7 | scorers implemented + smoke-tested | ✅ | run 2026-07-09 (GPUs 0/2/3) | done |
| C1–C7 | drop-noisy retrains (10 keep-sets) | ✅ | **AUM + PVI win; drop-hard hurts** (table below) | 🟢 **surgical denoise CONFIRMED** |
| §4 | inference gate (MSP per tier, AUROC, risk-coverage) | ✅ | AUROC-hard ~0.83 for base AND denoised; win = easy-tier F1 +0.010 | gate inherited, NOT improved — no routing lever |
| champ | pvi06+aum06 × champion recipe (E8a+LS richargs full_data) | 🏃 | launched 2026-07-09 GPU2/3 (`e22_*_champion`) | → held-out check, then submit |

### Retrain results (raw uncal val Macro-F1; granite v1, standard split; run 2026-07-09)
Where each method cuts, on the data map: [datamap_drops](figures/datamap_drops.png) ·
[datamap_pvi](figures/datamap_pvi.png) (`plot_drop_maps.py`).
In-pipeline **baseline `coreset_base` (full 56k) = 0.7498** (historical E9 control 0.7458). Δ vs 0.7498.

| keep-set (method) | drop | val Macro-F1 | Δ base | verdict |
|---|---|---|---|---|
| **pvi06** (C7 PVI) | 6% | **0.7561** | **+0.0063** | 🥇 best — info-theoretic mislabel |
| **aum06** (C2 AUM) | 6% | **0.7546** | **+0.0048** | 🥈 margin mislabel |
| **aum15** (C2 AUM) | 15% | **0.7542** | **+0.0044** | ✅ robust to drop level |
| **pvi15** (C7 PVI) | 15% | **0.7514** | **+0.0016** | ✅ still positive |
| forget (C5 forgetting) | never-learned | 0.7505 | +0.0007 | ~flat |
| cl (C1 cleanlab) | ~self-conf | 0.7500 | +0.0002 | ~flat |
| el2n06 (C6 EL2N) | 6% | 0.7458 | −0.0040 | ❌ hurts |
| cart06 (C4 cartography drophard) | 6% | 0.7435 | −0.0063 | ❌ drop-hard hurts |
| cart15 (C4 cartography drophard) | 15% | 0.7419 | −0.0079 | ❌ worse |
| el2n15 (C6 EL2N) | 15% | 0.7368 | −0.0130 | ❌ worst |

## §4 · Gate analysis — can the denoised model RULE OUT hard samples at inference? (2026-07-09)

**Question (user):** beyond the +F1 headline, does denoise training make the model's own confidence a
better inference-time gate — confident on easy, unconfident on hard? `gate_analysis.py`, raw fp32
logits (NO calibration), qwen3-ruler tiers (easy 10714 · ambig 1900 · suspect-mislabel 1386), logits
cached `analysis/cache/coreset_gate_logits.npz`. Figures: [msp_tiers](figures/gate_msp_tiers.png) ·
[risk_coverage](figures/gate_risk_coverage.png) · [suspect_behavior](figures/gate_suspect_behavior.png).

**Verdict: the gate is INHERITED, not improved — denoising buys easy-tier accuracy, not hard-sample
detection.** AUROC(MSP → ruler-hard): base 0.830 · pvi06 0.835 · aum06 0.833 (Δ ≈ noise); AUROC(own-error)
even dips (0.845 → 0.836). Consistent with the E20 model-independent ~0.85 gate ceiling.

| model | overall F1 | easy F1 | ambig F1 | suspect F1 | MSP easy/ambig/suspect | AUROC hard / own-err | gate F1 @76.5% cov (oracle) |
|---|---|---|---|---|---|---|---|
| base | 0.7500 | 0.9107 | 0.4239 | 0.1767 | 0.875 / 0.527 / 0.769 | 0.830 / 0.845 | 0.8054 (0.9107) |
| pvi06 | 0.7565 | **0.9204** | 0.3929 | 0.1577 | 0.890 / 0.516 / 0.790 | 0.835 / 0.836 | 0.8050 (0.9204) |
| aum06 | 0.7545 | **0.9209** | 0.3935 | 0.1599 | 0.899 / 0.520 / 0.797 | 0.833 / 0.841 | 0.8024 (0.9209) |
| cl | 0.7501 | 0.9178 | 0.4112 | 0.1529 | 0.982 / 0.908 / 0.967 ⚠️ | 0.767 / 0.787 | 0.8090 † |
| cart15 | 0.7430 | 0.8960 | 0.4100 | 0.1457 | 0.858 / 0.555 / 0.739 | 0.817 / 0.835 | 0.7805 (0.8960) |

Readings: (1) **easy-tier gain is where the denoise win lives** (+0.010 easy F1 for pvi06/aum06); suspect-tier
F1 *drops* — the expected §3 confound (better model penalized by wrong labels). (2) **Suspect tier confirmed
as mislabels:** ALL models (incl. base) put 81–85% of suspect predictions on the qwen3-consensus class, only
9–12% on the label — the models already predict the true class; the label disagrees. Denoising doesn't change
behavior on hard rows, it stops *learning from* them. (3) **MSP-gate ≪ oracle** (0.805 vs 0.92 @76.5% cov):
confidence and ruler-hardness overlap imperfectly; no routing lever here (E6/E20 again). (4) ⚠️ `cl` trains to
extreme overconfidence (MSP ≈ 0.97–0.98 on *every* tier) → worst detector (0.767); its risk-coverage curve
(† and the violet line in the figure) is inflated by massive MSP ties at 1.0 — actual coverage > nominal — do
not read it as a gate win. (5) ambig-tier MSP stays low (~0.52) for base/pvi06/aum06 — genuine-difficulty rows
remain visibly uncertain; the *detectable* hard population is the ambiguous one, not the mislabeled one.

**Verdict:** **PVI (C7) and AUM (C2) — information-theoretic / margin-based mislabel scorers — are the clear
winners** (+0.004–0.006, positive at BOTH 6% and 15% drop). Cleanlab (C1) and forgetting (C5) ~flat.
**Cartography-drophard (C4) and EL2N (C6) HURT, worse the more aggressive** — exactly the §1 prediction:
surgical *mislabel* removal helps modestly; blanket *drop-hard* starves rare/ambiguous signal. Best coreset
gain **+0.0063 (pvi06)** on granite screen. **Per-§3 tier read (coreset_eval.py → `eval_results.md`):** on the
**label-trustworthy subset the win is larger & cleaner — clean-val macro-F1 +0.011 (pvi06 0.8762 vs base 0.8656),
easy-tier +0.009** — while hard-tier F1 drops (0.238 vs 0.267 base), which is the *expected* confound (suspect-mislabel
labels penalize a model that predicts the TRUE class, §3). So denoise is **real and stronger than the +0.006 headline**.
Next = confirm pvi06/aum06 on the qwen3 champion. Method column maps to §2 (C1 cleanlab · C2 AUM · C4 cartography · C5 forgetting ·
C6 EL2N · C7 PVI).
