# E32 · Training-time augmentation / noise — `research/augmentation`

**Status: ✅ DONE — CLOSED NEGATIVE (2026-07-13, vast 44640537). All 5 runs finished.
NOTHING beat champion 0.7803: adversarial/consistency noise HURTS (FGM ε1.0 0.7712 /
R-Drop 0.7716 / FGM ε0.5 0.7734), gentle noise FLAT (NEFTune 0.7794 / hist_dropout
0.7786). No arm promoted. Logits for all 5 cached (e26_screen_logits_s3.npz). Instance
stopped (not destroyed). ⚠️ GPU1 was faulty (blacklisted) — caused false "R-Drop
unstable"/"FGM ε-bracket" NaN scares mid-run; corrected. See Verdict + GPU1 note.**

---

## E36 · Data-augmentation revisit (2026-07-13) — the FOUR untried directions

User reopened augmentation ("try data augmentation approach", all four directions). Scoping
first, then arm A run. Venue for training arms = **vast.ai** (user call); this box (ArchServer
RTX 4060 8GB) does the free local screens only.

### Sub-sequence / prefix expansion — CLOSED on data analysis (NOT run)
The task is next-action prediction over a session; "sub-sequence expansion" = turn one
length-N session into N examples via its prefixes (`[A₁..A_{k-1}] → A_k`). **But the
organizers already did this**: the 70k train rows ARE the per-step prefixes (`…-step_08` =
prefix through step 7 → predict step 8). Measured yield of expanding anyway: **70,000 rows →
73,205 recoverable prefix-points = only +4.6% new** (3,205 gap-fill rows, e.g. sessions
missing an intermediate `step_12`). The only *other* thing prefixes add is different history
*truncations* of the same target — which is `hist_dropout`/truncation-jitter, already
FLAT-to-negative (E32 hist_dropout, E24 token-drop −0.008). ⇒ near no-op; closed without a run.
(Script: ad-hoc analysis, `/home/kyusang/.conda/envs/dacon/bin/python`.)

### Arm A · Inference-time multi-view averaging — ❌ FLAT (best Δ +0.0003, noise)
The professor's LITERAL idea (feed multiple augmented views, average outputs), never run
before (E32 = training-time regularizers; E23 = entropy-min TTA). One trained model → serialize
each val row into K views → softmax-average → argmax. Two screens on the honest 14k val
(`split_indices seed=42`), fp16, local 4060:

**Screen 1 — honest model `e9_granite_ls` (v1+LS, split-trained → clean val), degraded views**
`CMD: CUDA_VISIBLE_DEVICES=0 python -m experiments.augmentation.e36_multiview_screen` (default ckpt).

| View | macro-F1 | Δ vs train view |
|------|----------|-----------------|
| v1 full (**train view = baseline**) | **0.7558** | — |
| v1 h12 | 0.7558 | +0.0000 (histories rarely >12 events → no diversity) |
| v1 h8 / h6 / h4 | 0.7359 / 0.7006 / 0.6617 | −0.020 / −0.055 / −0.094 (truncation destroys signal) |
| nometa / leanact (full) | 0.7185 / 0.6972 | −0.037 / −0.061 (off-variant, off-distribution) |
| **avg trunc[full,12,8]** | **0.7560** | **+0.0001** (best; noise) |
| avg trunc[full,12,8,6] / +[..,4] | 0.7477 / 0.7389 | −0.008 / −0.017 |

**Screen 2 (decisive) — champion `e8a_ls_richargs` fed EQUAL-QUALITY close-cousin views**
richargs/richfiles/richmeta all ≈0.797 (E25: near-identical quality, different framing) — the
genuine "diverse-but-equal" views. Val is full_data-contaminated (absolute F1 inflated ~+0.017
vs the 0.7803 CV) but near-equal views memorize comparably ⇒ the *relative* avg-vs-single signal
is clean. `CMD: CUDA_VISIBLE_DEVICES=0 python -m experiments.augmentation.e36b_richviews`.

| View / combo | macro-F1 | Δ vs richargs |
|--------------|----------|---------------|
| richargs full (**baseline**) | **0.7974** | — |
| richfiles / richmeta / richargs-h12 | 0.7969 / 0.7962 / 0.7974 | −0.0005 / −0.0012 / +0.0000 |
| **avg[argfull,filesfull,metafull]** | **0.7977** | **+0.0003** (best; noise) |
| avg pairs / avg-all-4 | 0.7970–0.7974 | −0.0004 … +0.0000 |

**Verdict (arm A ❌):** inference-time multi-view averaging does NOT help — neither degraded
views (−) nor equal-quality diverse views (+0.0003 noise). **Mechanism:** the +0.008 E26 trio
gain is *different-model* error-diversity; one model's outputs across input *views* are too
CORRELATED (same weights → same mistakes) to average profitably. The view change doesn't
decorrelate errors the way a different seed/backbone does. Figure:
[figures/e36a_multiview.png](figures/e36a_multiview.png). Probs cached
`experiments/augmentation/e36_multiview_probs.npz`. Contamination caveat only weakens the
already-null screen-2; screen 1 is honest and equally flat. **Not submission-worthy; not an
ensemble-member play.**

#### Arm A follow-up (user 2026-07-13) · PROGRESSIVE-HISTORY averaging — ❌ net-negative
User asked to try the specific view axis: per row build hist0, hist1, …, histH (hist_j = last
j history events, hist0 = zero history) and average outputs over j=0..H. Ran on both models,
14k val, `CMD: CUDA_VISIBLE_DEVICES=0 python -m experiments.augmentation.e36c_progressive_hist`
(detached setsid; log `experiments/augmentation/e36c_progressive.log`).

**Single-view F1 is MONOTONIC in depth** (e9 honest / e8a champ): hist0 0.4044/0.4331 → hist3
0.6541/0.6928 → hist6 0.7006/0.7409 → hist9 0.7446/0.7932 → hist12(full) **0.7558/0.7974**.
More history is strictly better ⇒ every truncated view is a *handicapped* prediction, not a peer.

| Averaging scheme | e9 honest Δ | e8a champ Δ |
|---|---|---|
| single full-history (baseline) | 0.7558 | 0.7974 |
| avg hist0..histH (literal) | −0.0616 | −0.0523 |
| avg hist1..histH (drop hist0) | −0.0480 | −0.0447 |
| avg histceil(H/2)..histH | −0.0153 | −0.0131 |
| avg last-3 depths {H−2..H} | +0.0003 | −0.0043 |

**Verdict:** confirmed net-negative — the more truncated depths you average in, the worse
(−0.05..−0.06 for the full sweep); only the top-3-depths scheme reaches noise (+0.0003/−0.0043)
because histories cap at 12 so hist10–12 barely differ. Distinct failure mode from the rich-view
screen: there views were equal-quality-but-redundant (Δ≈0); here they're a strict QUALITY LADDER
(deleting history deletes signal — E1/E24 low-redundancy input). Both models agree exactly.
Figure: [figures/e36c_progressive_hist.png](figures/e36c_progressive_hist.png). Script
`e36c_progressive_hist.py`.

#### Arm A probe (user idea) · FORCE history length = 12 by padding — ❌ hurts
Hypothesis: is the monotonic depth curve information content (1) or a length/positional bias
(2, in which case padding short rows to 12 would help)? Pad each row's history to 12 events at
the FRONT (real recent events kept adjacent to prompt); padding adds NO real info (dup/random)
so this isolates length-bias. Honest e9, clean 14k val (9,633/14,000 rows have <12 real events).
`CMD: CUDA_VISIBLE_DEVICES=0 python -m experiments.augmentation.e36d_padhist` (log e36d_padhist.log).

| Mode | overall | Δ | short rows H<12 | Δ short |
|---|---|---|---|---|
| natural (baseline) | 0.7558 | — | 0.7464 | — |
| pad-tile (cycle row's own history) | 0.7045 | −0.0513 | 0.6682 | −0.0783 |
| pad-repfirst (repeat oldest event) | 0.7257 | −0.0301 | 0.7050 | −0.0414 |
| pad-random (events from other rows) | 0.7057 | −0.0502 | 0.6717 | −0.0747 |

**Verdict: hypothesis (2) refuted — it's information content, not length bias.** Every padding
mode HURTS (−0.03 to −0.05 overall, −0.04 to −0.08 on the affected short rows). Duplicating a
row's real history (tile/repfirst) creates repetitive out-of-distribution inputs the model
misreads; random events inject false context. (Aside: short-real-history rows score HIGHER
naturally, 0.7464, than the H=12 group 0.7158 — long capped sessions are the harder population;
the within-row truncation monotonicity ≠ across-row difficulty.) No length knob to exploit.
Figure: [figures/e36d_padhist.png](figures/e36d_padhist.png). Script `e36d_padhist.py`.

— Remaining arms (A2 train-on-mixed-views · C mixup · D E15c
offline-translation) NOT run (user: stop after A). A2's prior is now weak: for multi-view to
pay, the model must be TRAINED so views are equal-quality-AND-decorrelated, and screen-2 shows
even equal-quality views stay error-correlated under shared weights.

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
| A1 | **stage 1** | FGM adversarial perturbation on the embedding layer (Miyato et al.) — **doubles as E34-A**: one run, result cross-recorded on the [E34 adversarial-axis board](../adversarial/results.md) | ε=1.0 | ✅ **implemented 2026-07-13** — `--fgm_eps` (`make_fgm_trainer`) | **0.7750 · +0.0017 vs anchor 0.7733** (2026-07-13, vast). Marginal — member candidate, sub-gate. ⚠ AWP (E34-C) beat it +0.0054 → AWP is the axis pick, not FGM |
| A2 | **running (healthy)** | R-Drop: two dropout passes + symmetric KL (official `dropreg/R-Drop`) | α=1.0, LS kept in CE term, bs2×accum8 (co-located w/ B1 on GPU3) | code ✅. ⚠️ First two attempts NaN'd on the **faulty GPU1** (not R-Drop). Re-run on healthy GPU3 **trains fine** (finite loss ~22 incl. KL term, no divergence) → **R-Drop is NOT unstable**; the "needs KL-warmup" note was wrong | — |
| A3 | done | NEFTune-style uniform embedding noise | α=5 | ✅ implemented 2026-07-13 — `--neftune_alpha` → HF Trainer built-in `neftune_noise_alpha` (official neelsjain/NEFTune hook) | **0.7794** (full_data CV, GPU2) · −0.0009 vs champion 0.7803, +0.0004 vs from-scratch anchor 0.7790 → **flat, sub-gate** |
| B1 | done (filler) | `--hist_dropout` (per-epoch random history-event drop) | p=0.1 single probe | flag already existed (`serialize()` + dynamic path `src/finetune.py:1144`), never screened | **0.7786** (full_data CV, GPU3) · −0.0017 vs champion, −0.0004 vs anchor 0.7790 → **flat, sub-gate** |
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

**Stage-1 run 2026-07-13 (vast instance 44640537, 4× RTX 3090, champion recipe granite
richargs `--full_data --loss ls 0.1 --epochs 3 --lr 2e-5 --bs4×accum4 --max_len 512
--seed 42`, one lever per GPU). `/venv/main/bin/python -m src.finetune`.** Results
auto-rsynced to `output/pat/ft_..._<tag>/` on per-arm completion.

Anchors: champion full_data CV **0.7803**; from-scratch full-input anchor **0.7790**
(E24). Gate = ≥ +0.003. These are 3500-slice full_data CV — a screen, not LB-rankable
(E8/E26: slice mis-ranks — LB judges finals).

| Run | Lever | GPU | full_data CV | vs champ 0.7803 | vs anchor 0.7790 | verdict |
|-----|-------|-----|--------------|------------------|-------------------|---------|
| A3 NEFTune | `--neftune_alpha 5` | 2 | **0.7794** | −0.0009 | +0.0004 | ✅ flat, sub-gate |
| B1 hist_dropout | `--hist_dropout 0.1` | 3 | **0.7786** | −0.0017 | −0.0004 | ✅ flat, sub-gate |
| s2 FGM ε=0.5 | `--fgm_eps 0.5` | 2 | **0.7734** | −0.0069 | −0.0056 | ✅ **HURTS** (less than ε=1.0) |
| A2 R-Drop α=1.0 | `--rdrop 1.0` (bs2×accum8) | 3 | **0.7716** | −0.0087 | −0.0074 | ✅ **HURTS** |
| A1 FGM ε=1.0 | `--fgm_eps 1.0` | 0 | **0.7712** | −0.0091 | −0.0078 | ✅ **HURTS** (worst) |

### ✅ VERDICT (all 5 done 2026-07-13) — Track A embedding-space augmentation is NET-NEGATIVE / flat for this task

**Nothing beat the champion (0.7803) or even the from-scratch anchor (0.7790).** Two clusters:
- **Adversarial / consistency noise HURTS ~−0.007 to −0.009:** FGM ε=1.0 **0.7712**, R-Drop
  α=1.0 **0.7716**, FGM ε=0.5 **0.7734**. The FGM ε sweep is **monotonic toward ε=0**
  (0.5 hurts less than 1.0) → the optimum is *no* adversarial perturbation. R-Drop's KL
  consistency term likewise degrades.
- **Gentle noise is FLAT:** NEFTune α=5 **0.7794**, hist_dropout 0.1 **0.7786** — within
  noise of the anchor, neither near the +0.003 gate.

**Interpretation:** the input is low-redundancy (E24) and the labels weakly semantic; adding
noise — in embedding space (FGM/R-Drop/NEFTune) or history structure (hist_dropout) — either
does nothing or actively erases signal. The professor's "average over augmented inputs" idea,
adapted to this task, does **not** help under the champion recipe. **No arm promoted; no
ensemble-member candidate** (all sub-anchor). E32 CLOSED negative. (Stage-2/3 moot — no arm
cleared the gate.) ⚠️ GPU1 fault (below) caused false intermediate NaN scares; corrected.

**Logits:** all 5 models' held-out logits extracted on the local 4060 (fp32, 3.5k held-out,
richargs) → `analysis/cache/e26_screen_logits_s3.npz` (`extract_e32_logits.py`); self-checks
match training within fp16-save rounding. Available for later ensemble search, though all
sub-anchor. Instance 44640537 **stopped** (not destroyed) after harvest.
⚠️ Note: the A1 arms-table row carries a **cross-recorded FGM number from the shared
E34 adversarial run** (0.7750 vs E34's own anchor 0.7733) — different anchor than E32's;
E32's own FGM full_data CV lands when A1 above finishes. AWP (E34-C) reportedly beat FGM
on that axis (see [E34 board](../adversarial/results.md)).

### ⚠️ FAULTY GPU1 on vast instance 44640537 (2026-07-13)

**GPU1 produces NaN gradients on any training run**, while GPU0/2/3 are fine. Isolated
by a controlled test: **FGM ε=1.0, the exact config running healthy on GPU0, NaN'd
immediately on GPU1** (loss 9.2e4/grad nan) — same code, seed, hyperparameters, only the
GPU differs. Basic matmul on GPU1 passes (no ECC on consumer 3090s), so the fault is
load-dependent. **Every "divergence" attributed to a method was actually GPU1:**
- R-Drop α=1.0 & α=0.5 (GPU1) → NaN — **NOT a real R-Drop instability; being re-tested.**
- FGM ε=2.0 & ε=0.5 (GPU1) → NaN — **the ε≤1.0 "bracket" is INVALID; ε=1.0 (GPU0) is the
  only valid FGM point so far.**
GPU1 blacklisted; usable GPUs on this box = 0/2/3 only.

Numbers land when the runs finish (full_data CV vs champion 0.7803). —
