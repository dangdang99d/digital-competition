# E32 · Training-time augmentation / noise — `research/augmentation`

**Status: 🔲 QUEUED (user 2026-07-13). Code ready + smoke-tested ✅; stage-1 trainings not yet dispatched.**

## Motivation

Professor's suggestion: feed multiple augmented versions of each input and average the
outputs. The literal text-space version is closed for this task:

- **Translation** — E15a NO-GO (NLLB-600M 93 min for 30k on a local 3090 ≫ 10-min budget);
  training on translated data also failed (teammate's attempt + our audit).
- **Paraphrase / synonym-style augmentation** — labels are weakly semantic, so
  "meaning-preserving" rewrites are not guaranteed label-preserving.
- **Token masking / deletion** — E24: input is low-redundancy; even a 10% token drop cost
  −0.008 macro-F1.

What remains — and what this experiment tests — is noise injected where language doesn't
matter: **embedding space (Track A, PRIMARY)**. All arms are training-time regularizers:
zero inference cost, packaging and server timing unchanged. The "averaged outputs" half of
the professor's idea is already banked by E26 model ensembling (trio LB 0.78719, +0.008);
a winning noised model additionally becomes a diverse-regime member candidate for that pool.

**Track B (structure-level) demoted — user call 2026-07-13, expected minimal benefit.**
Supporting evidence: E24-B's "lossless but no gain" cuts both ways (meta fields the model
barely uses are shortcuts not worth starving); the trio already banks view diversity via
its e25c_richmeta member, so single-model view mixing plausibly lands *between* the two
view optima instead of above them. B3 dropped outright; B2 parked unimplemented; B1 kept
only as an idle-GPU filler because it needs zero code.

## Arms

| Arm | Priority | Lever | Config (stage 1) | Impl status | Result (Δ vs anchor) |
|-----|----------|-------|------------------|-------------|----------------------|
| A1 | **stage 1** | FGM adversarial perturbation on the embedding layer (Miyato et al.) — **doubles as E34-A**: one run, result cross-recorded on the [E34 adversarial-axis board](../adversarial/results.md), where it gates the PGD/AWP escalations | ε=1.0 | ✅ **implemented 2026-07-13** — `--fgm_eps` (`make_fgm_trainer`, wraps the final trainer_cls so both passes inherit the recipe's loss); **smoke ✅** (see below) | — |
| A2 | **stage 1** | R-Drop: two dropout passes + symmetric KL (official `dropreg/R-Drop`) | α=1.0, LS kept in CE term | ✅ **fixed 2026-07-13** — `make_aux_trainer` gained `ce_mode="ls"` (E30-style: LS inside ALL training CE terms — first pass, second pass, weighted); the `--loss ls` × aux assert relaxed to ce/ls; default `ce_mode="ce"` byte-identical to before; **smoke ✅** (see below) | — |
| A3 | gated | NEFTune-style uniform embedding noise | α=5 | ✅ **implemented 2026-07-13** — `--neftune_alpha` passes through to HF Trainer's built-in `neftune_noise_alpha` (transformers 4.51.3 ships the official neelsjain/NEFTune hook verbatim — reference-impl invariant satisfied by construction) | — |
| B1 | filler | `--hist_dropout` (per-epoch random history-event drop) | p=0.1 single probe | **flag already exists** (`serialize()` + dynamic path `src/finetune.py:1144`), never screened — idle-GPU filler only (user: minimal expected benefit) | — |
| B2 | ⏸ parked | meta-subfield dropout + history-truncation jitter | — | do NOT implement unless Track A wins and the combo stage wants a data-side partner | — |
| B3 | ❌ dropped | ~~view mixing: richargs↔richmeta per epoch~~ | — | dropped (user 2026-07-13): trio already holds view diversity (e25c member); mixing risks landing between the view optima | — |

## Implementation (2026-07-13, code ready + smoke-tested — stage 1 not dispatched)

All changes in `src/finetune.py`, additive and default-off (shared-tree invariant).

**Smoke ✅ (2026-07-13, local ArchServer RTX 4060, conda dacon, transformers 4.51.3):**
one tiny run per arm — granite-311m richargs `--loss ls --precision bf16 --max_len 128
--batch_size 4 --epochs 0.01` + the arm's flag (`--fgm_eps 1.0` / `--rdrop 1.0` /
`--neftune_alpha 5`), logs `<scratchpad>/e32_smoke_a{1,2,3}.log`. All three: exit 0,
lever log line fired ("FGM adversarial training eps=1.0" / "aux CE terms use label
smoothing eps=0.1" / "NEFTune alpha=5.0"), eval sane (val F1 .035/.056/.063 · eval_loss
2.49–2.54 after 140 tiny steps), artifact saved + reloads, NEFTune hook confirmed absent
from the saved model. Train-summary loss ≈10.7 in ALL arms incl. non-FGM = pre-existing
grad_accum×4 display quirk, not instability. ⚠ smoke ≠ recipe validation: max_len 128 /
0.01 epochs exercises code paths only; FGM's ~2× step cost will show at 512, not here.

- **A1 `--fgm_eps`** — `make_fgm_trainer` overrides `training_step`: clean pass via
  `super()`, perturb the input-embedding weight by `eps·g/‖g‖₂`, adversarial
  forward/backward with loss normalization mirroring `Trainer.training_step`
  (v4.51.3 source fetched and matched), exact clone-restore. Wraps the FINAL
  trainer_cls → composes with `--loss ls` (and `--rdrop`); asserted off for
  `--distill_from`/LTP. Reference: Miyato et al. official `adversarial_text`
  formula; port = the standard PyTorch FGM class (weight-matrix perturbation);
  divergence from the TF original (per-example input perturbation) documented in
  the docstring. Direction is invariant to fp16 grad scaling; skips (with one
  warning) if embeddings are frozen.
- **A2 `--rdrop` + `--loss ls`** — `make_aux_trainer(ce_mode, label_smoothing)`:
  when `ce_mode="ls"`, all three training CE terms (first pass, rdrop second pass,
  hard-boundary weighted) use `label_smoothing`; eval path stays plain CE. The
  blocking assert now permits ce/ls (focal/wce/la still excluded). Loss shape
  matches official `dropreg/R-Drop`: `0.5·(CE₁+CE₂) + α·½[KL(p₁‖p₂)+KL(p₂‖p₁)]`
  (our KL uses `batchmean`; constant-factor differences vs sum-reduction repos are
  absorbed by the α sweep).
- **A3 `--neftune_alpha`** — pass-through to `TrainingArguments.neftune_noise_alpha`;
  HF 4.51.3's hook is the official NEFTune implementation (train-only, removed at
  eval/save automatically).

## Protocol

- **Recipe:** champion E8a+LS richargs full_data bf16 + exactly ONE lever per arm,
  **from scratch** — never warm-start (E24 recovery trap). All arms share the same
  from-scratch anchor and eval slice.
- **Baseline:** champion full_data CV **0.7803**. Eval always un-augmented
  (`hist_dropout` docstring: never use for eval). Raw uncalibrated logits, macro-F1.
- **Stages:** 1 — A1 + A2 (2 trainings); 2 — sweep the strength knob (ε / α) of
  gate-clearing arms, add A3 if signal; 3 (optional) — combine winners. B1 may run
  opportunistically on an otherwise-idle GPU at any point.
- **Gates:** promote at ≥ +0.003 vs 0.7803. Marginal +0.001–0.003 → ensemble-member
  candidate (diverse training regime), judged by honest OOF / LB, never the slice
  (E8/E26: slice mis-ranks — LB judges any final claim).
- **Code changes:** additive and default-off only (shared-tree invariant); log exact
  commands with every result; figure per result.

## Results

—
