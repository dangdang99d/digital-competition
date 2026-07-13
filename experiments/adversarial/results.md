# E34 · Adversarial-training escalation — PGD + AWP — `research/adversarial`

**Status: ✅ STAGE 1 DONE + AWP LB-CONFIRMED 2026-07-13. AWP single-model → 🥇 LB 0.78557 =
NEW SINGLE-MODEL SOTA (+0.0040 vs prev best single E28 t043fd 0.78155; +0.0082 vs champion
0.77738); a single model TIES the 2-granite ensemble pairs (0.78548) at 5:10. PGD hurts
−0.0060 (closed), FGM marginal +0.0017. The axis inverted its own prior (E13-discounted AWP
won, recommended PGD lost). AWP now = prime ensemble-member candidate. See Results.**

## Motivation

Adversarial training perturbs the training signal in the direction that most increases the
loss and asks the model to be correct anyway — an untried axis here until E32-A1 (FGM).
The three standard methods differ only in *what* is perturbed and *how carefully*:

| Method | Perturbs | Inner max | Step cost | Arm |
|---|---|---|---|---|
| **FGM** | embedding weight | 1 linear step `ε·g/‖g‖` | 2× | **E34-A** (executes as **E32-A1**; result cross-recorded here) |
| **PGD-K** | embedding weight | K projected steps in an ε-ball | (K+1)× | **E34-B** |
| **AWP** | model weights | relative-norm step `γ·‖w‖·ĝ` | ~2.5× | **E34-C** |

E34 is the canonical record for the **whole adversarial axis**. Arm A (FGM) physically
runs inside E32 (it doubles as that experiment's embedding-noise arm A1) — ONE training
run, recorded in both boards; E34-B/C are the escalations, specified up front so they
aren't re-derived later. All arms are training-time only: zero inference cost, packaging
and server timing unchanged; a winner lifts the single model AND is a diverse-regime
member candidate for the ensemble pool (different training dynamics ⇒ different errors,
the currency E26/E29 showed uniform averaging converts).

## Arms

| Arm | Lever | Stage-1 config | Prior | Impl status | Result (Δ vs anchor / vs FGM) |
|-----|-------|----------------|-------|-------------|-------------------------------|
| A | **FGM** — one-shot embedding perturbation `ε·g/‖g‖`, second forward/backward, restore (Miyato et al.) — **= E32-A1, single run double-recorded** | ε=1.0 | ✅ axis opener: cheapest, one knob | ✅ implemented (`--fgm_eps`, `make_fgm_trainer`) | **0.7750 · +0.0017** vs anchor → marginal (member candidate, sub-gate) |
| B | **PGD-K** — iterated FGM: K small steps, project `‖δ‖₂ ≤ ε` after each; drops the one-step local-linearity assumption | K=3 · α=0.3 · ε=1.0 (embedding weight, same hook as FGM) | ✅ if FGM shows signal (same mechanism, sharper max) | ✅ implemented (`--pgd_eps/--pgd_alpha/--pgd_k`, `make_pgd_trainer`) | **0.7673 · −0.0060** vs anchor → ❌ HURTS, close |
| C | **AWP** — perturb *weights* toward worst case, gradient at `w+v`, restore, step; explicit flat-minima training | γ=1e-3 · adv-lr 1e-4 · enable from epoch 2 (`--awp_start_epoch 1.0`) · all `*weight*` tensors | ➖ **E13 SAM closed** (same flat-minima family; near-zero run variance = no flatness headroom) | ✅ implemented (`--awp_gamma/--awp_lr/--awp_start_epoch`, `make_awp_trainer`) | 🥇 **0.7804 · +0.0071** vs anchor → PROMOTE (LB probe + member) |

## Protocol

- **Recipe/anchor:** from-scratch champion recipe (E8a: granite · richargs · LS ε=0.1 ·
  full_data · bf16) vs the **same from-scratch anchor shared with E32** — one anchor for
  the whole noise family keeps arms comparable; never warm-start (E24 trap). If the E28
  `--full_data` promotion validates on LB first, recipe AND anchor switch to t043 together.
- **Screen split:** honest session-grouped fold (3.5k slice mis-ranks — E8/E26 lesson);
  LB judges any promotion.
- **Cost:** A ≈ 4× step time (~5h/3090 run) · B ≈ 2.5× (~3h).

## Gates (design — see Results for how the run actually resolved them)

- **Arm A (FGM)** dispatches with E32 stage 1 — no gate of its own; its result gates B/C.
- **Dispatch B/C (from arm A):** FGM ≥ +0.003 vs anchor → run B+C · FGM ∈ (0, +0.003) →
  B only (sharper inner max may rescue a weak signal; C's flatness lever already
  discounted by E13) · FGM ≤ 0 → **close B/C unrun**, axis dead.
- **Promotion:** escalation arm > FGM by ≥ +0.002 → replaces FGM as axis pick, joins the
  member pool (judged on honest OOF, then LB) · arm ≈ FGM → keep FGM (cheaper, one knob) ·
  both ≤ FGM → close; FGM is the axis's final word.
- **What actually happened:** all 3 arms + anchor ran in parallel (4 idle GPUs → compute
  free → the dispatch gate became moot; ran everything, gated on *promotion* instead). The
  outcome inverted the design's priors — see Results.

## Results

**Stage 1 — 2026-07-13, vast `E28_optuna_r1` (4×3090, one arm/GPU), from-scratch champion
recipe (granite E8a: richargs · LS ε=0.1 · full_data · bf16 · lr 2e-5 · 3ep · eff-batch 16
via E19 fast shape · max_len 512 · seed/init_seed 42 · fp16 save · best-epoch RAM snapshot,
no disk checkpoints). All exit 0. Metric = best-epoch val macro-F1 on the full_data 3.5k
holdout (⚠ this slice mis-ranks — E8/E26 — so LB judges any promotion).**

| Arm | lever | val macro-F1 | Δ vs anchor | best epoch | verdict |
|-----|-------|:---:|:---:|:---:|---|
| anchor | champion, no adversarial | 0.7733 | — | 3 | in-environment control |
| A · FGM | `--fgm_eps 1.0` | 0.7750 | **+0.0017** | 3 | marginal — member candidate, sub-gate |
| B · PGD | `--pgd_eps 1.0 --pgd_alpha 0.3 --pgd_k 3` | 0.7673 | **−0.0060** | 3 | ❌ HURTS → close |
| C · **AWP** | `--awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0` | **0.7804** | **+0.0071** | 3 | 🥇 **LB 0.78557 = single-model SOTA** (5:10); promoted |

**Read-out — the axis inverted its own prior.** The design expected PGD (recommended
escalation) to win and AWP (E13-discounted flat-minima family) to be the long shot. The
opposite happened: **AWP won by +0.0071** — the best single-lever result since LS ε=0.1
(E9, +0.0107) — while **PGD actively hurt** (−0.0060; the iterated ε-ball attack
over-regularizes at K=3/α=0.3/ε=1.0) and **FGM landed marginal** (+0.0017). So AWP > FGM by
+0.0054 (clears the "replaces FGM as axis pick" +0.002 bar) and clears the +0.003
promotion gate outright. E13's "no flatness headroom" prior is **refuted for AWP** — weight
perturbation found headroom SAM's zero-variance analysis said wasn't there (likely because
AWP attacks per-tensor relative-norm, a different geometry than SAM's global step).

- **Anchor note:** in-environment anchor = 0.7733, **−0.007 below the documented champion
  0.7803** — attributable to the E19 fast shape (`--batch_size 16 --grad_accum 1
  --group_by_length` changes batch composition vs the original 4×4) and/or torch.compile
  numerics on this box. Running the shared anchor was load-bearing: gating AWP against the
  documented 0.7803 would have shown +0.0001 (noise) instead of the true +0.0071 lift.
  AWP at 0.7804 *matches* the documented champion despite the recipe's own baseline
  sitting lower — i.e. it recovers the fast-shape's lost 0.007 and is a genuine gain over
  the fast-shape champion it's built on.

**LB CONFIRMED (2026-07-13):** `submit_0713_awp.zip` (601M, single-model granite AWP, same
richargs `script.py` as champion) → **LB 0.78557, inference 5:10**. This is a **new
single-model SOTA**: +0.0040 vs the previous best single (E28 t043fd 0.78155), +0.0082 vs
champion granite_ls 0.77738, +0.0063 vs qwen3_ls 0.77921. A SINGLE model **ties the 2-granite
ensemble pairs** (pair A 0.78548, pair B 0.78498) and sits just under the trio (0.78719) — at
5:10, faster than the trio's 6:52. The slice **under-predicted** as expected (slice 0.7804 →
LB 0.78557, +0.005; same direction as every E8/E26 slice→LB gap), so the +0.0071 slice Δ was
if anything conservative. E13's flatness-headroom refutation now holds on the LB, not just the
slice.

**Next:** (1) 🎯 **AWP into the ensemble** — it is a strong, diverse-regime member (weight-space
adversarial vs the E28 models' HP variation); swapping it into / adding it to the trio via the
E30 honest-OOF screen is the clearest path past the 0.78916 team best. (2) AWP ε/γ stage-2
sweep (γ∈{5e-4,2e-3}, start-epoch∈{0,2}) — obvious now that the lever is LB-validated. (3)
FGM (+0.0017) is a secondary member candidate. All 4 models on NFS
`output/e34/vast_r1/{e34_anchor,e34_a_fgm,e34_b_pgd,e34_c_awp}/` (fp16, 629M). Instance
`E28_optuna_r1` (44568954) **stopped** (storage-only) after harvest.

## Implementation (2026-07-13, code ready + smoke-tested — stage 1 not dispatched)

Additive, default-off flags in `src/finetune.py`, colocated with `make_fgm_trainer`;
`py_compile` clean. All three adversarial levers mutually exclusive (hard assert:
"escalations are compared, not stacked") and asserted off for `--distill_from` / LTP.

**Smoke ✅ (2026-07-13, local ArchServer RTX 4060 8GB, conda dacon — E32 smoke protocol):**
granite-311m richargs `--loss ls --precision bf16 --max_len 128 --batch_size 4
--epochs 0.01` + the arm's flags, logs `<scratchpad>/e34_smoke_{b,c}.log`. Both: exit 0,
lever log line fired ("PGD adversarial training eps=1.0 alpha=0.3 k=3" / "AWP … gamma=0.001
adv_lr=0.0001 from epoch 0.0"), eval sane (val F1 .026/.058, eval_loss 2.50–2.51 ≈ E32's
2.49–2.54), best-epoch snapshot + artifact saved; step cost visibly ~4×/~2× (67s / 63s vs
E32's arms at the same shape). Train-summary loss ≈10.5 = the known grad_accum×4 display
quirk. Two smoke-only accommodations for the 8GB card (real runs = 3090 24GB, unaffected):
`--optim adafactor` (AdamW's 2.5GB states leave no room for embedding-sized transients)
and, for AWP, `--awp_start_epoch 0` (the 1.0 default never activates inside a 0.01-epoch
smoke — REMEMBER: any short validation run must override it).

- **B `--pgd_eps / --pgd_alpha / --pgd_k`** — `make_pgd_trainer`: K perturb-steps on the
  input-embedding weight, ε-ball (Frobenius) projection after each; final adversarial
  backward ADDS to the untouched clean grads; exact clone-restore; loss normalization
  mirrors `Trainer.training_step` (v4.51.3), reused from the FGM port. **Documented
  divergence from the reference PGD class:** intermediate attack directions come from
  `torch.autograd.grad(adv_loss, emb)` instead of the backup-grads/zero/backward/restore
  dance — semantically identical (final grad = clean + adv-at-worst-point; --grad_accum
  untouched by construction) and avoids a model-size grad backup per step (the reference
  mechanics OOM'd the 8GB smoke card; emb-sized temps are `del`-ed before each adversarial
  forward for the same reason).
- **C `--awp_gamma / --awp_lr / --awp_start_epoch`** — `make_awp_trainer`: per step (once
  `state.epoch ≥ awp_start_epoch`), every trainable `*weight*` tensor steps toward its
  gradient by `adv_lr·‖w‖·g/(‖g‖+1e-6)`, clamped elementwise into `w0 ± γ|w0|`; one
  adversarial forward/backward adds to the clean grad; exact restore. Port of the standard
  Kaggle AWP class (Feedback-Prize lineage, `adv_param="weight"` name filter, attack after
  clean backward / restore after adversarial backward). ~1 model-size backup per step.
- Both wrap the FINAL trainer_cls (same slot as FGM) → compose with `--loss ls` /
  `--rdrop`; reported loss stays the CLEAN loss.

## Results log

(append: date · arm · key numbers · memory file. Arm A numbers land here AND in
[augmentation/results.md](../augmentation/results.md) — one run, two boards.)

- **2026-07-13 · stage 1 (all arms + anchor, vast E28_optuna_r1 4×3090)** — anchor 0.7733 ·
  FGM 0.7750 (+0.0017) · **AWP 0.7804 (+0.0071) 🥇** · PGD 0.7673 (−0.0060). AWP promotes;
  PGD closes; FGM marginal member candidate. Axis inverted its prior (AWP beat PGD, refuting
  E13's flatness-headroom claim for weight perturbation). Models on NFS `output/e34/vast_r1/`.
  Memory: [[adversarial-awp-wins]].
- **2026-07-13 · AWP LB** — `submit_0713_awp.zip` → **LB 0.78557, 5:10 = new single-model
  SOTA** (+0.0040 vs E28 t043fd 0.78155; +0.0082 vs champion). Single model ties the 2-granite
  pairs (0.78548). Slice under-predicted (0.7804→0.78557). Next: AWP → ensemble member.
