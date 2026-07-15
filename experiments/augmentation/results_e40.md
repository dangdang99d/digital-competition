# E40 · Training-time char-level text augmentation (free-text only) — `research/augmentation`

**Status: 🟢 CODE READY + smoke-passed (2026-07-13). Not dispatched — awaits vast GO.** Code lives
in `experiments/augmentation/` (`text_noise.py`, `run_e40.py`); `src/` untouched. Venue = vast.ai.

## Implementation (2026-07-13, code ready + smoke-passed)

- **`text_noise.py`** — language-aware noiser. `noise_text(s, rng, p)`: KO Hangul → jamo perturb
  (decompose via 0xAC00 formula → change 1 of initial/medial/final → recompose; confusion tables
  for vowel/tense-consonant, batchim drop/add/alter), ASCII letter → swap/delete/insert/QWERTY-sub,
  digits/punct/symbols untouched. `noise_sample_freetext(sample, rng, p)` copies + noises ONLY
  `current_prompt` + USER-event `content` (no deepcopy; non-user events reused verbatim).
- **`run_e40.py`** — driver: champion recipe (richargs+LS+full_data+bf16+3ep+lr2e-5+**bs16+accum1**),
  injects noise by REPLACING `src.finetune.build_dynamic_dataset` in-process with a version that
  char-noises free-text per `__getitem__` (α = fraction noised each epoch); routes train to the
  dynamic path via a `--hist_dropout 1.0` **sentinel** (value ignored — history is NOT dropped).
  Val uses the clean static path (`build_dataset`) → eval always clean. `--p --alpha` knobs.

**Smoke ✅ (local 4060, p=0.02 α=0.5, bs4/len128/limit400/0.05ep):** exits 0; the noising dataset
installs and logs (`E40 char-noise TRAIN dataset: p=0.02 alpha=0.5 … history NOT dropped`), train +
clean eval run, artifact saved. Field/full-string examples verified: KO jamo typos (기능→기느,
작업→적업, 없나→헚나), EN char ops (transform→teransform, cases→caaes), and **all structure lines
(meta + ACTION) byte-identical** (structure-preservation check = True). ⚠ smoke = code path only.
⚠ the finetune log emits `history dropout p=1.0` (the sentinel) just above the E40 line — cosmetic.

**Run (per arm, vast):** `python -m experiments.augmentation.run_e40 --p <p> --alpha <a>`
→ `output/pat/ft_..._e40_p<p>_a<a>` + `ft_results_e40.csv`.

## Motivation

User idea (2026-07-13): character-level noise as a **training-time surface-robustness
regularizer** — perturb the raw characters of each sample during training so the model sees
varied spellings, then eval clean. Distinct from what's closed: E24 = token *drop* (info loss,
−0.008), E32 = *embedding*-space noise (FGM/NEFTune/R-Drop, flat-to-negative). Char-level surface
noise on the **text** has not been tried.

**Scope decision (user): FREE-TEXT FIELDS ONLY** — noise applies to `current_prompt` and `USER:`
message content, and NOTHING else. Metadata header, action names, args, verdicts, structure, and
labels stay byte-exact. This is the principled test: noising the structured tokens the model keys
on is the E24 −0.008 trap; noising only the natural-language part tests genuine surface robustness.

**Data facts driving the design (measured):** free-text is **mixed Korean (60%) / English (40%)**
and **code-heavy technical chat** (`useAuth`, `refreshToken`, `PermissionDenied`, `403`→`302`,
`pyproject`). ⇒ char noise is chosen because it is **language-agnostic** (works on KO+EN alike, no
lexical resource) — unlike synonym replacement (needs bilingual thesaurus + code-identifier
guarding; deferred) and contextual/MLM aug (user: too difficult). ⚠️ Caveat: code identifiers
embedded in prose (e.g. `useAuth`) WILL get noised under free-text scope — that is the surface-
robustness premise; a token-protecting variant is a fallback if v1 hurts.

## Prior (expectations modest — sharpened by lit search 2026-07-13)

Not data-limited (70k rows; ceiling is label ambiguity — E13 near-zero variance, E33/KD negative,
~40% irreducible ambiguity E22), and the test set is CLEAN (no typos), so training on noised text
is a pure regularizer. E32's gentle embedding noise (NEFTune) was FLAT; char noise may land similar.

**Literature (search 2026-07-13) — mechanism supported, clean-test upside is small:**
- **Char-noise aug's documented benefit is ROBUSTNESS to noisy input, ~NEUTRAL on clean test**
  ([Understanding Model Robustness to User-generated Noisy Texts, arXiv 2110.07428]; [Belinkov &
  Bisk, ICLR 2018]). Our test is clean → literature predicts ~neutral, not a clean-accuracy boost.
- **Train on a MIX of clean + noised**, not all-noised (Belinkov & Bisk) → motivates the per-sample
  augment-rate knob α below (keep a clean fraction each epoch).
- **Char noise heavily disrupts SUBWORD tokenization** ([Continual Pre-training on Char-level Noisy
  Texts, TACL]) → a small p already re-segments the input a lot; center the sweep LOW.
- **Jamo-level is the validated granularity for Korean** classification ([Sub-Character Architecture
  for Korean, arXiv 1707.06341]; [Char-level Embedding in KO Sentence Classification, arXiv
  1905.13656]) → confirms the jamo design below.
- **Gains concentrate in low-resource; smallest on classification** ([To Augment or Not to Augment?,
  MIT Press 2022]; [Text Data Aug for Korean, MDPI 2022] classification gain 0.08%) → 70k + classi-
  fication = limited headroom.

**Net:** mechanism is sound but the clean-macro-F1 upside is likely neutral-to-marginal; the honest
payoff (noise robustness) isn't scored on a clean test. ⇒ run it as a **tight sweep with a hard
gate**, not a campaign. Bar: beat the champion cls baseline by ≥ +0.003 on clean macro-F1.

## Arms — two knobs: p (per-char noise) × α (per-sample mix rate); else = champion recipe

Two knobs, both lit-motivated: **p** = per-character noise prob within free-text (kept LOW — subword
tokenization is sensitive); **α** = fraction of samples noised each epoch (the clean+noised MIX; the
rest pass through clean). Dynamic per-epoch → over epochs every sample is seen both clean and noised.

| Arm | p (per-char) | α (mix rate) | Note |
|-----|-------------|--------------|------|
| A0 | — | 0 | **control** = champion cls from-scratch (shared anchor) |
| A1 | 0.02 | 0.5 | gentle noise, half-clean mix — **primary bet** (lit favors low p + mix) |
| A2 | 0.05 | 0.5 | moderate noise, half-clean mix |
| A3 | 0.02 | 1.0 | gentle noise, ALL samples noised — isolates the mix (α) effect vs A1 |

Stop-rule: if A1 & A2 are both ≤ anchor, don't sweep further (matches the low-headroom prior). If
A1 signals (≥ +0.001), expand p∈{0.01,0.03} × α∈{0.3,0.7} around it.

**Char ops are LANGUAGE-AWARE, routed per character** (free-text is 60% KO / 40% EN — a single
ASCII-style op set is wrong for Hangul: whole-syllable substitution 학→가 is a word-level change,
not a typo). Each char in a free-text span, with probability p:
- **Hangul syllable → JAMO-level perturbation** (the correct char-level noise for Korean):
  decompose via the standard formula (`0xAC00 + 초성×588 + 중성×28 + 종성`, 19×21×28 — dependency-
  free) → change ONE component → recompose. Realistic single-keystroke typos: (1) vowel confusion
  ㅏ↔ㅓ / ㅗ↔ㅜ / ㅐ↔ㅔ (keyboard/phonetic neighbors), (2) batchim(받침) drop/add/alter — the most
  common real KO typo (작업→자업, 같이→같시), (3) tense/aspirated consonant confusion ㄱ↔ㄲ↔ㅋ /
  ㅈ↔ㅉ↔ㅊ / ㅂ↔ㅃ↔ㅍ. Keeps the word recognizable (small visual/phonetic edit).
- **ASCII letter → char op**: adjacent-swap · delete · insert · QWERTY-adjacent substitute.
- **digits / punctuation / code symbols → left alone**.

One `p` knob over both. **Dynamic per-epoch** (re-noise each epoch, like `hist_dropout`'s dynamic
path) so the model sees fresh surface variation; **eval always clean**. (Alternative framing —
*static* offline expansion into original + K noised copies, the literal "increase number of data"
— is a fallback if dynamic signals but caps prevent enough exposure.)

## Protocol

- **Recipe:** champion granite richargs + LS 0.1 + full_data + bf16(auto) + 3 ep + lr 2e-5 +
  **bs16 + grad_accum 1** + max_len 512 + seed 42, **from scratch**, ONE lever = char-noise p.
- **Baseline:** A0 no-aug = champion cls from-scratch (its own anchor); champion full_data CV
  **0.7803** for context. Raw uncalibrated logits, macro-F1.
- **Read-out (primary):** clean full_data CV per arm vs A0 (screen, not LB-rankable — E8/E26 slice
  mis-ranks). This is the gate.
- **Read-out (secondary, diagnostic):** also eval each arm on a NOISED val (same jamo/ASCII noiser,
  fixed seed, p=0.05) — makes a flat clean result interpretable: if noised-val robustness rises while
  clean is flat, the noise "worked" but the clean test just can't reward it (matches the lit); if
  neither moves, the noise did nothing. Diagnostic only — never the gate (test is clean).
- **Gates:** promote ≥ **+0.003** vs A0; marginal +0.001–0.003 → ensemble-member candidate (a
  noise-regularized model is a diverse regime); judged by honest OOF / LB, never the slice.
- **Implementation invariants (when built):** code in `experiments/augmentation/` (E40 files)
  ONLY; `src/` untouched. Augmentation hooks the text/serialization step **in-process** (wrap the
  raw-sample free-text before `serialize()`, or the dynamic dataset), default-off so no other run
  changes. Noise touches ONLY `current_prompt` + `USER:` content; a unit check must assert the
  serialized string is byte-identical to champion outside those spans when p=0.
- **Figure:** p-sweep curve (macro-F1 vs p) with the A0 baseline line.

## Results

### Diagnostic (pre-training) — how much does the noise hurt the BEST model? (2026-07-13)
Apply the E40 noiser to the VAL data (all rows, fixed seed), eval best single model = AWP (E34,
LB 0.78557, richargs, val* full_data-contaminated) + honest e9 (v1, clean val) cross-check.
`CMD: CUDA_VISIBLE_DEVICES=0 python -m experiments.augmentation.e40_val_robustness`.

Eval slice = the **honest full_data 3.5k held-out** (`train_test_split(va, .25, seed=42)`), the
slice every `--full_data` run (AWP incl.) held out — so AWP is uncontaminated here (an earlier run
on the full 14k read AWP 0.8055, inflated by full_data training-on-val; corrected below).

| p | AWP macroF1 (Δ) | AWP KO / EN | e9 honest (Δ) |
|---|---|---|---|
| 0.0 | **0.7810** | 0.7808 / 0.7790 | 0.7493 |
| 0.02 | 0.7758 (−0.0052) | 0.7782 / 0.7687 | 0.7429 (−0.0064) |
| 0.05 | 0.7637 (−0.0173) | 0.7711 / 0.7449 | 0.7331 (−0.0162) |
| 0.10 | 0.7385 (−0.0425) | 0.7534 / 0.7029 | 0.7049 (−0.0444) |

**Read:** smooth dose-response, **both models agree** (AWP honest 0.7810 ≈ its LB/CV level). The
noise is REAL (measurable effect), gentle at p=0.02 (~−0.005), moderate at 0.05 (~−0.018). Notable:
**English degrades MORE than Korean** at higher p (AWP p=.10: KO 0.7725 vs EN 0.7307) — the jamo
single-component edit keeps KO words recognizable, while ASCII delete/swap mangles short EN words
more → the noise is *not* strength-matched across languages (KO effectively gentler at equal p).
Figure: [figures/e40_val_robustness.png](figures/e40_val_robustness.png), data `e40_val_robustness.json`.
**Implication:** this is the robustness BASELINE E40 training would try to move; it confirms the noise
does something but says nothing yet about clean-test gain (lit prior: ~neutral). Training arms (A0–A3)
not yet run.
