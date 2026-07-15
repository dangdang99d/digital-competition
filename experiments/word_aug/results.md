# E39 · Offline word-level augmentation — `research/word-aug`

**Status: 🔴 CLOSED — NEGATIVE (2026-07-15). Offline word-aug does not help this task: the clean
flagship generator (Qwen3-30B) scored no better than noisy C-MLM, both ~−0.02 vs A0.** Code in `experiments/word_aug/`
(`src/` untouched). The word-level escalation of E38 (char-noise): where E38 perturbs surface
characters as a training regularizer, E39 generates *new labeled rows* by rewriting the free-text.

## Motivation

E38 tested char-level noise (KO jamo + EN ASCII). E39 goes up one level — **word-level rewriting**
to add training data. From the lit survey (2026-07-13), the four word-level families ranked for
this bilingual, code-heavy, weak-semantic-label task:

- **Contextual MLM substitution** (Kobayashi 2018; Conditional-BERT/C-MLM, Wu et al. 2019, arXiv
  1812.06705) — a masked-LM replaces words with context-fitting alternatives. Best classic method.
- **LLM paraphrase** (AugGPT; LLMs-vs-classic, arXiv 2408.16502) — ~100% valid samples, highest
  quality, code-identifier-aware if prompted.
- **Back-translation** (arXiv 1903.09244) — strong low-resource, but *"on the full dataset neither
  augmentation improves upon SOTA."*
- **Synonym/EDA** (arXiv 1901.11196; KoEDA) — "improvements minimal", breaks on word-sense + named
  entities. Worst fit (our text is full of code identifiers). **Dropped.**

We run the top two: **B1 = C-MLM (XLM-RoBERTa-large, masked-word substitution)** · **B2 = LLM
paraphrase (`gpt-4o-mini` via the OpenAI Batch API)**. B2 was originally scoped as a local
Qwen2.5-14B via vLLM; switched to the API for reliability + quality + cost (batch ≈ $6 for all
138k rows, and the API preserves code identities the local path did not).

## Prior (from lit) — modest, but a real angle

Word-level aug **vanishes with data** (multiple sources; back-translation explicit) and we have
70k. BUT the metric is **macro-F1** and classes are imbalanced **8.8×** (`edit_file` 8937 vs
`web_search` 1018 on the 56k train) — rare classes are the low-resource pockets *inside* a
high-resource set. So the bet is **class-balancing augmentation**, which directly targets macro-F1
and dodges the high-resource ceiling. Two hazards the papers name — named-entity/identifier
corruption and word-sense errors — are handled by code-token protection + strict "preserve intent".

## Design (user-locked 2026-07-13)

- **Split = session-fold-0** (leak-free StratifiedGroupKFold, NOT the session-leaky
  `split_indices`): 56k train / 14k holdout. Chosen so **A0 = the E38 t070 fold-F1 0.7806 is
  reused directly as the control** (already trained; no GPU wasted). Augment ONLY from the fold-0
  train (verified pool∩holdout = 0 → no leak); eval = the fold-0 holdout, directly comparable to
  0.7806. (Earlier draft used the 56k/14k `split_indices`; superseded by session-fold-0 to match
  t070 + stay leak-free.)
- **Equal aug volume (user: "69k+69k")** so balanced-vs-blanket isolates TARGETING, not volume:
  - **balanced** → lift every class up to the max (8,937): **69,118 copies** from 27,228 rows
    (rare rows get multiple distinct paraphrases). Post-aug class range = 8937–8937 (flat).
  - **blanket** → **69,118 copies** sampled UNIFORMLY from the pool (39,708 rows) — auto-matched
    to balanced's volume.
- **Free-text only:** rewrite `current_prompt` + `USER:` content; structure/action-names/args/
  metadata/labels byte-exact. C-MLM protects **code identifiers** (camelCase/snake_case/digit/path/
  ALLCAPS/backtick never masked); the LLM is *instructed* to copy identifiers verbatim.
- **Two independent C-MLM sets** (seed 42 + seed 43) generated → gives 2× data for an optional
  volume test (run only if a 1× balanced arm shows promise — user call).

## Baseline model = **E38 t070** (AWP-Optuna best; user-locked 2026-07-13)

**A0 control = t070** — the best AWP-Optuna trial (`experiments/optuna/results_e38.md`),
richargs + AWP-tuned recipe on **session-fold-0** (leak-free StratifiedGroupKFold), **fold-F1
= 0.7806**. This number is REUSED as A0: t070 is *already trained* (56k-fold weights pending
pull), so **we do NOT retrain the control** (user call — no GPU wasted on A0). Aug arms are
`Δ vs 0.7806` + per-rare-class F1.

**Why t070, not granite_ls:** granite_ls (LB 0.77738) is not our best single model. t070 is the
tuned peak of the AWP axis (E34 AWP blind-defaults = LB 0.78557 SOTA; E38 tuned it to fold 0.7806).
Testing augmentation on the deployment-grade recipe avoids the E22 trap (a data-gain on a weak
recipe vanishing under the strong one). Aug (data) ⟂ AWP (weights) → they stack.

**t070 recipe** (E39 arms train this FROM SCRATCH + aug): granite richargs · LS 0.0518 · lr
2.932e-5 · **eff_batch 16 (bs16×accum1)** · warmup 0.1462 · wd 3.2e-3 · awp_gamma 3.421e-3 ·
awp_lr 2.063e-4 · awp_start 1 · epochs 4 (best-epoch) · **`--session_fold 0`** · init_seed 42.

**Split = session-fold-0** (NOT the session-LEAKY split_indices): 56k train / 14k leak-free
holdout. Augment ONLY from the fold-0 train (verified pool∩holdout = 0). Eval = the fold-0
holdout, directly comparable to t070's 0.7806.

Single-model context (LB): granite_ls 0.77738 · optuna-t043 0.78155 · **AWP 0.78557** ·
trio(ensemble) 0.78780. (t070's 0.7806 is a fold-F1; its LB comes from the `t070fd` full_data
promotion, separate.)

## Arms — factorial {generator} × {targeting}; A0 REUSED = **4 trainings** (not 5)

A0 is not retrained (= t070 fold 0.7806). The 4 aug arms train the t070 recipe from scratch on
fold-0 train + aug, one per GPU (single wave on 4×3090). Compares generator (C-MLM vs LLM) AND
targeting (balanced vs blanket) at equal 69k volume.

| Arm | generator | targeting | train rows | run? |
|-----|-----------|-----------|-----------|------|
| A0 | — (control = **t070**) | none | 56k | ✅ reuse 0.7806 |
| B1-bal | C-MLM (XLM-R-large) | balanced 69k | 56k+69k | ❌ 0.7627 (−0.0179) |
| B1-bln | C-MLM | blanket 69k | 56k+69k | ❌ 0.7638 (−0.0168) |
| B3-bal | **LLM Qwen3-30B-A3B-2507-FP8** (clean flagship) | balanced 66k | 56k+66k | ❌ 0.7582 (−0.0224) |
| B3-bln | LLM Qwen3-30B | blanket 66k | 56k+66k | ❌ 0.7639 (−0.0167) |
| B2-bal/bln | LLM (`gpt-4o-mini`) | bal/bln | — | ⊘ dropped (gen ~26%, never trained — B3 already settled it) |

(B2 gpt-4o-mini is a *weaker* clean generator than B3 Qwen3-30B; once B3 failed, B2 could not change the verdict, so it was not run to conclusion.)

Compares generator quality (C-MLM vs LLM) AND targeting (balanced vs blanket) at equal volume.
Optional 2nd wave (only if a balanced arm signals): **2× volume** using the seed42+seed43 C-MLM sets.

## Models · GPUs · volume

| | model | where | throughput |
|---|-------|-----|-----------|
| C-MLM gen | `FacebookAI/xlm-roberta-large` (560M, fill-mask) | vast, 20-worker fleet across 4×3090 | ~110 rows/s → 138k in ~20m |
| LLM gen | **`gpt-4o-mini`** (OpenAI **Batch API**, −50%) | API (no GPU) | 3 batches ×≤50k, async; ~$6 total |
| each training | **t070 recipe** (granite richargs+AWP, **bs16×1**, from scratch) | vast, 1 arm/GPU | AWP ~2× |

Generated volume: C-MLM 2 seeds × {bal,bln} × 69k + LLM {bal,bln} × 69k.
**Data provenance / backups (local):** `output/e39/cmlm_v2_maskp10/` (seed42) ·
`output/e39/cmlm_v2b_seed43/` (seed43) · `output/e39/openai_batch/` (batch inputs + LLM outputs).

## Protocol & gates

- Train the **t070 recipe** (granite richargs + AWP-tuned, bs16 grad_accum 1, from scratch,
  `--session_fold 0`, init_seed 42) on `fold-0 train (56k) ∪ aug`; eval on the fold-0 holdout
  (14k). Verified injection: `train=125,118` (56k+69,118 aug), `val=14,000` (no aug). Raw uncal
  macro-F1.
- **Validate first (user-gated):** QC generated rows vs originals for label preservation before
  training. Done: C-MLM 100% structure/label preserved, code identifiers kept; GPT-4o-mini 8/8
  JSON, natural meaning-preserving paraphrases, identifiers verbatim (see Results/QC).
- **Gates:** overall macro-F1 ≥ **+0.003** vs A0 (0.7806) promotes; ALSO report **per-rare-class
  F1** (the targeted win should surface there even if overall is flat). Fold screen → LB confirms.
- **Expectation:** balanced = the real bet (rare-class F1 ↑); blanket likely flat (high-resource).

## Implementation (`experiments/word_aug/`, `src/` untouched)

- **`common.py`** — fold-0 pool (`session_fold_indices`), code-token protection (`is_code_token`),
  free-text span extract/replace, `select_targets(balanced|blanket)` with auto-matched 69k volume.
  ✅ verified (pool 56000; bal/bln both 69,118 copies; post-balance 8937–8937; pool∩holdout = 0).
- **`gen_cmlm.py`** — XLM-R mask→predict on non-code words. **mask_p = 0.10** (EDA "sweet spot";
  the paper shows α>0.1 *hurts* — 0.15 was the BERT-*pretraining* convention, wrong for aug),
  replacements filtered to real word tokens (no punctuation/garbage), spans with <4 candidate
  words skipped, per-text batched forward, `--shard/--nshards` data-parallel, incremental write.
- **`gen_llm.py` / `openai_batch.py`** — LLM paraphrase. Prompt: system rule *"rephrase, keep exact
  intent, same language, copy every code identifier/path/number/backtick VERBATIM, return a JSON
  array (same length/order)."* `openai_batch.py` builds ≤50k-request batch files → uploads →
  creates batches → `fetch` parses outputs by `custom_id={tgt}|{idx}|{copy}` → augmented JSONL.
  (`gen_llm.py` is the local-vLLM variant, retired.)
- **`run_e39.py`** — unions augmented JSONL into training in-process: extends `load_samples`, and
  `session_fold_indices` splits ORIGINALS into fold-0 train/holdout then appends every aug index
  to train → aug never enters the holdout. Recipe = t070 (AWP). `src/` untouched.

## Results

### QC (pre-training, user-gated) — 2026-07-13
- **C-MLM** (mask_p 0.10, 69,118 rows): structure byte-identical **100%**, label==source **100%**,
  free-text changed **100%**, code identifiers preserved. ⚠ Inherent **word-sense/entity drift**
  (the papers' known MLM failure): e.g. `rayon`→`꾸준히`, `마이그레이션`(migration) dropped — some
  label-relevant. Garbage-token fills (`~`) fixed. This is the "noisy classic method" arm.
- **GPT-4o-mini** (8-sample test): **8/8 JSON OK**, natural meaning-preserving paraphrases (KO+EN),
  **all code identifiers verbatim** (`Button.tsx`, `--config`, `db.ts`, `tsconfig.json`, `N+1`,
  `src/data/augment.py`). Clearly higher fidelity than C-MLM — the reason the LLM arm exists.

### Training

**C-MLM (B1) — ❌ REJECTED (−0.017 vs A0).** Both arms trained the t070 recipe from scratch on
fold-0 train (56k) ∪ 69,118 C-MLM copies (injection verified train=125,118 / val=14,000),
best-epoch on the leak-free fold-0 holdout:

| arm | targeting | ep1 | ep2 (best) | ep3 | ep4 | best-epoch | Δ vs A0 (0.7806) |
|-----|-----------|-----|-----|-----|-----|-----------|------|
| A0 (t070) | — | | | | | **0.7806** | — |
| B1-bal | balanced 69k | 0.7427 | **0.7627** | 0.7544 | 0.7517 | 0.7627 | **−0.0179** |
| B1-bln | blanket 69k | 0.7379 | **0.7638** | 0.7531 | 0.7520 | 0.7638 | **−0.0168** |

Both peak at epoch 2 then decline (overfit-to-noise curve); **targeting is irrelevant**
(balanced ≈ blanket, Δ 0.0011). The class-balancing bet does not pay: C-MLM's inherent
word-sense / named-entity drift (QC: `rayon`→`꾸준히`, dropped `마이그레이션`) injects label noise
that swamps any rare-class gain — the papers' known MLM-aug failure mode, confirmed on this
weak-semantic-label task. Consistent with the E22 prior (LS already absorbs label noise; adding
noisy copies only hurts). This isolates the *generator*: whether augmentation itself is dead, or
only the noisy generator is, is decided by the clean LLM arm.

**LLM Qwen3-30B (B3) — ❌ REJECTED (−0.017 to −0.022 vs A0). The decisive arm.** The clean flagship
generator (`Qwen3-30B-A3B-Instruct-2507-FP8`, served FP8 on 5090s) produced high-fidelity
paraphrases — QC on the 66k balanced set: **label 100%, structure 100%, Korean-stays-Korean 98.1%,
code identifiers verbatim 99.5%** (vs C-MLM's entity drift). Trained the t070 recipe from scratch on
fold-0 train (56k) ∪ 66k Qwen copies, best-epoch on the fold-0 holdout:

| arm | targeting | ep1 | ep2 | ep3 | ep4 | best-epoch | Δ vs A0 (0.7806) |
|-----|-----------|-----|-----|-----|-----|-----------|------|
| B3-bal | balanced 66k | 0.7271 | 0.7567 | **0.7582** | 0.7540 | 0.7582 | **−0.0224** |
| B3-bln | blanket 66k | 0.7295 | **0.7639** | 0.7590 | 0.7518 | 0.7639 | **−0.0167** |

**Clean generator ≈ noisy generator** (B3 0.7582/0.7639 vs B1 0.7627/0.7638) — the near-perfect
paraphrase fidelity bought *nothing*. This isolates the cause: it is **not generator quality;
offline word-augmentation itself does not help this task.** A faithful paraphrase carries the same
weak-semantic label ambiguity (the real ceiling, cf. [[E37]] label-ambiguity result), so extra
copies only re-add the noise floor. Targeting is again irrelevant (balanced ≈ blanket across *both*
generators) → the class-balancing bet does not pay.

**Annealing variant** (2 epochs aug+clean → 2 epochs clean-only, single continuous LR schedule via
`run_e39_anneal.py`) was scoped as a rescue (turn aug off at the end to undo the ep-4 overfit dip).
Not pursued to conclusion — given both plain-aug generators land ~−0.02 with no sign the aug epochs
add useful signal, the prior is recovery *toward* baseline but not past 0.7806 (label-ambiguity, not
distribution-shift, is the ceiling). Left as the one open lever if word-aug is ever revisited.

### Verdict — E39 CLOSED, negative
Offline word-level augmentation (generate new labeled rows by rewriting free-text) does **not**
improve macro-F1 on this task, across the full generator-quality axis: noisy C-MLM (−0.017), clean
flagship Qwen3-30B (−0.017 to −0.022). Both fail identically; **generator quality is not the
lever**. Class-balanced ≈ blanket in every arm → the rare-class-balancing hypothesis is refuted.
Consistent with the lit prior (word-aug vanishes at high-resource; we have 70k) and the E22/E37
priors (LS absorbs label noise; the ceiling is label ambiguity). Do not re-propose offline
paraphrase/word augmentation as an accuracy play here. Data + trained weights: `output/e39/`.
