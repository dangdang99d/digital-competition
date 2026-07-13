# Branch: noise-robust — robust losses + sample selection vs the LS champion

Logical group `research/noise-robust` (committed on the working branch; the per-branch
`research/*` topology was planned but never created — all work lives on the shared tree, see
[EXPERIMENTS.md](../EXPERIMENTS.md)). All scores **raw uncalibrated macro-F1** (project
invariant — NO calibration).

## Objective

The task carries ~20% label noise. Test whether **dedicated noise-robust training methods** —
robust losses and dual-network sample selection — **beat or stack with LS ε=0.1**, the champion
loss (E9 +0.0107). E9's loss screen only covered CE/focal/LS/wce/la; the canonical noisy-labels
toolkit (GCE, SCE, APL, bootstrap, ELR, co-teaching, DivideMix) is untested here.

**The bar is beat LS, not beat CE.** LS is itself a noise-robust regularizer (it caps the
per-label gradient), and E22 proved a CE-recipe win is a mirage — its denoise gains vanished
under LS. So every arm is judged against LS 0.7565, with CE 0.7458 alongside only for
attribution ("worse LS" vs "real signal").

## Overlap with prior work — what's genuinely new (confirmed 2026-07-13)

Checked against the two adjacent lanes before scoping:

| E35 method | Tried before? | Evidence |
|---|:--:|---|
| GCE / SCE / NCE+RCE (APL) / bootstrap | ❌ **No** | E9 loss screen covered **only** CE / focal / LS / wce / la — none of the robust-loss family |
| ELR | ❌ **No** | Not a loss E9 tried; not the data-selection mechanism E22 used |
| co-teaching / JoCoR / DivideMix (Stage C) | ⚠️ **Scoped, never run** | **= E22's "Group 3"** (co-teaching / MentorNet / DivideMix / SELFIE), ⛔ gated on a denoise gain that wasn't met → never run |

**Stage A/B are the genuinely-untested core.** E9 was *imbalance*-motivated (not noise), E22 was *data-centric* (score rows → drop → retrain — a different mechanism from reshaping the loss). Stage C is **not new** — it is [E22](../coreset/results.md)'s never-run Group 3 by name; E35 owns it as an explicit **revival**, still gated, not a fresh idea.

**⚠️ E22 is a strong NEGATIVE prior for this whole lane.** E22 §4: data denoising helped the CE recipe (+0.005–0.006) but **failed under the LS champion** (pvi06 **−0.0103**), because **LS already absorbs the label noise** (soft targets damp mislabel gradients). GCE/SCE/APL/ELR are all *adaptive LS* — a second noise-handling mechanism with, per E22, little left to absorb once LS is present. So expect **match-or-lose against the LS bar** (like la τ=1 was −0.0002); the standalone-vs-CE screen is the only place E22 saw noise-handling room. **Implication for the gate order:** screen Stage A/B on the **CE recipe first** (where noise-handling demonstrably has headroom), LS-confirm only survivors; treat Stage C as likely dead-on-arrival under LS.

## What we already know about the noise (from E22 §1 — don't re-derive)

- Of the ~23.5% val error: **~6% is clean two-model-consensus mislabel** (removable), **~40% of
  errors are low-confidence boundary ambiguity** (irreducible — no loss beats Bayes error), and
  the noise **concentrates on 4 synonymous file-ops** (`list_directory` / `read_file` /
  `glob_pattern` / `grep_search`). → the noise is **asymmetric / class-conditional**, not uniform.
- **E22 denoising fails under LS** (−0.004 to −0.010 on the champion recipe): LS absorbs the
  mislabel gradients, so static data-cleaning is redundant. Sample-selection methods (Stage C) are
  "denoise during training" and inherit this discount — hence they're gated, not run up front.

## Priors (be honest up front)

- **LS has won here; the label-softening family is strong.** GCE/SCE/APL are *adaptive* label
  softeners (they damp by the model's own confidence rather than smoothing uniformly). The open
  question is whether that adaptivity buys anything **on top of / instead of** LS's flat damping.
- **Residual error is largely intrinsic ambiguity** (ceiling/specialist probes saturate; reasoning
  caps ~0.28). Expect an **LS-magnitude gain at best, real chance of null** (like la τ=1 was
  −0.0002). This is a legitimate untested corner, not a headline lever.
- **Asymmetric noise → SCE / APL are more apt** than symmetric-noise-only framings; GCE is the
  general first swing; **ELR is the strongest single-network method** on 20%-noise benchmarks.
- **Bootstrap ≈ self-distillation** ("trust your own prediction") — overlaps the
  [self-distillation](../self-distillation/results.md) lane's soft end; note to avoid double-counting.
- **Granite screen is a FILTER, not a ranking** (E8 CV→LB reversal): a survivor must re-confirm on
  the champion recipe + qwen3 + LB before it's believed. LS itself transferred granite→qwen3.
- **Invariant — get the official code first.** Each arm below names the authors' repo; diff our
  implementation against it before trusting numbers (papers omit q/α/β defaults, EMA schedule,
  detach placement, warm-up epochs).

## Baseline / anchors (granite, standard 56k/14k split, from-scratch)

| Anchor | uncal macro-F1 | Role |
|---|:--:|---|
| granite **CE** control (E9) | **0.7458** | attribution floor — "beats plain CE at all?" |
| granite **LS ε=0.1** (E9) | **0.7565** | **THE bar** — "beats / stacks with the champion loss?" |
| champion E8a+LS richargs `--full_data` (transfer target) | 0.7803 (full_data CV) / ~0.7733 from-scratch in-env (E34) | survivors only; **LB judges** (full_data CV mis-ranks) |

Guardrails: **from-scratch, never warm-started** (E24 recovery trap: warm-start+3ep → ~0.763
regardless). Same seed/init as the anchors. Standard split, **not** `--full_data` (full_data CV is
contaminated for ranking — invariant).

## LS or not? (the training-recipe rule)

**Substitute losses (GCE / SCE / APL / bootstrap) train STANDALONE — no LS.** LS is not a neutral
base recipe here; it is a *competing* noise loss (target-softening). These four replace the CE/
target term, same slot LS occupies, so they are LS's rivals — bundling LS in would re-add its known
+0.0107 to every arm and mask each loss's own contribution. Read each against **both** anchors:
Δ-vs-CE = the loss's own effect (comparable to LS−CE=+0.0107); Δ-vs-LS = the decision (beat the
incumbent?). Bootstrap is itself *adaptive LS*, so +LS would be double-softening.

**ELR is the exception — it trains WITH LS (primary).** ELR *adds* an orthogonal regularizer
(`CE + λ·reg`) rather than replacing the target term, so it genuinely stacks: the deployment-natural
form is **LS-CE + λ·ELR-reg**, compared vs LS. (Also run ELR-on-CE vs CE to confirm the reg does
anything.)

**Stacking probe (secondary, low prior):** any substitute-loss survivor that beats/ties LS
standalone gets ONE `+LS` arm to test complementarity — expected redundant (E22: LS absorbs the
noise).

## Arms

Legend: ✅ win · ❌ closed · 🟡 marginal · ⛔ gated · 📋 planned (not run). "+LS?" = whether LS is
in the training recipe for that arm.

| Exp | Stage | Method | Venue | Net | +LS in recipe? | New code | Prior | Status | Screen |
|---|:--:|---|---|:--:|:--:|:--:|:--:|:--:|---|
| E35-A1 | A | **GCE** (q-sweep) | NeurIPS'18 | 1 | **No** (rival of LS) · +LS probe if survives | `--loss gce --gce_q` ✅ | general first swing | 🟢 **code ready** | — |
| E35-A2 | A | **SCE** (α,β) | ICCV'19 | 1 | **No** · +LS probe if survives | `--loss sce --sce_alpha/beta` ✅ | ✅ asymmetric-apt | 🟢 **code ready** | — |
| E35-A3 | A | **NCE+RCE (APL)** | ICML'20 | 1 | **No** · +LS probe if survives | `--loss apl --apl_alpha/beta` ✅ | ✅ principled | 🟢 **code ready** | — |
| E35-A4 | A | **Bootstrap** (soft/hard) | ICLR'15w | 1 | **No** (already adaptive-LS) | `--loss boot --boot_beta/mode` ✅ | ➖ ≈ self-distill | 🟢 **code ready** | — |
| E35-B  | B | **ELR** (λ, EMA β) | NeurIPS'20 | 1 | **Yes** (additive reg; LS-CE + λ·reg = primary) | `elr.py` ✅ | ✅ strongest 1-net | 🟢 **code ready** | — |
| E35-C1 | C | **Co-teaching** (= E22 Grp-3) | NeurIPS'18 | 2 | **No** · +LS probe if survives | `co_teaching.py` ✅ | ➖ ≈ E22 denoise | 🟢 **code ready** | — |
| E35-C2 | C | **JoCoR** (= E22 Grp-3) | CVPR'20 | 2 | **No** · co-reg agreement | `jocor.py` ✅ | ➖ ≈ E22 denoise | 🟢 **code ready** | — |
| E35-C3 | C | **DivideMix** (= E22 Grp-3) | ICLR'20 | 2 | **No** · semi-sup | `dividemix.py` ✅ | ➖ ≈ E22 denoise | 🟢 **code ready** | — |

**All 8 arms coded + smoke-tested end-to-end 2026-07-13** (user: "implement all") — but the GATES still
hold for RUNNING: C1 is user-un-gated; A→B→C ordering + the beat-LS bars below still decide what's worth
box time. Details + run commands in **§Implementation status** at the bottom.

**Stage C = [E22](../coreset/results.md) "Group 3" revival**, not a new idea — co-teaching/MentorNet/DivideMix/SELFIE were scoped there and ⛔ never run (gated on a denoise gain not met). E35 resurrects them only if Stage A/B beats the LS bar, i.e. only if the robust-loss angle finds noise-handling room E22's static denoising didn't.

**C1 implementation (✅ 2026-07-13, user un-gated Stage C directly):** `experiments/noise-robust/co_teaching.py`
— bespoke two-model loop (co-teaching doesn't fit HF `Trainer.compute_loss`), reuses `src.data`
(`load_samples`/`build_texts`/`split_indices`) + `src.finetune.build_dataset` + granite from HF base.
`loss_coteaching` is a faithful port of `bhanML/Co-teaching/loss.py` (per-sample CE → argsort →
keep `(1−R)` lowest → **exchange**: net1 updates on net2's picks, vice-versa); the official
`noise_or_not`/`pure_ratio` monitor is dropped (needs a ground-truth clean mask — synthetic-noise
only; ours is real/unknown). R(T)=`linspace(0, τ^c, num_gradual)` then flat τ. Recipe = the E9/E22
CE screen (standard split, v1, max_len 512, lr 2e-5, warmup 0.1, bf16 autocast). **Smoke-tested
end-to-end** (CPU + 4060: both nets, R-ramp, exchange, eval, CSV all execute; full training needs
a **≥16GB GPU** — two granite-311M + two AdamW states OOM an 8GB card, confirmed). Reports best-epoch
macro-F1 per net + Δ vs CE 0.7458 / LS 0.7565. **Run plan:** forget-rate sweep τ∈{0.06, 0.10, 0.20}
(E22 over-drop warning: aggressive drop HURT → low τ nearer the ~6% consensus-mislabel estimate),
epochs 5 / num_gradual 5, init_seed 42 vs 43. ⚠️ two-net diversity caveat (shared pretrained
backbone) is the key risk — see below.

## Method reference (what each does + official repo to diff against)

**Stage A — robust losses** (drop-in `classification_loss` modes; each tested **standalone vs LS**,
survivors also probed **stacked with LS**):

- **A1 · GCE** — `L_q = (1 − p_y^q)/q`, q∈(0,1]; interpolates CE (q→0) ↔ MAE (q=1). Gradient
  scales as `p_y^{q−1}`, so confident-wrong (noisy) samples get *shrinking* gradient. Sweep
  q∈{0.4, 0.7, 1.0}. Repo: Zhang & Sabuncu, *Generalized Cross Entropy* (ref impl
  `AlanChou/Truncated-Loss`, `HanxunH/Active-Passive-Losses`).
- **A2 · SCE** — `α·CE(p,y) + β·RCE(p,y)`; RCE = CE with prediction/label roles swapped (bounded,
  noise-tolerant). CE term keeps it learnable, RCE term robust. Grid α∈{0.1,1}, β∈{0.1,1}. Repo:
  `YisenWang/symmetric_cross_entropy_for_noisy_labels` (also `HanxunH/SCELoss-Reproduce`).
- **A3 · NCE+RCE (APL)** — Active-Passive Loss: a *normalized* active loss (NCE, maximizes p_y) +
  a passive loss (RCE/MAE, minimizes p elsewhere). Normalization → robust; active+passive pairing
  → still learnable. The most principled of the battery; strong on asymmetric noise. Repo:
  `HanxunH/Active-Passive-Losses` (official).
- **A4 · Bootstrap** — CE against a blended target `β·y_onehot + (1−β)·ŷ` (soft, β≈0.95) or its
  argmax (hard, β≈0.8): a confident model partially overrides a wrong label with its own belief.
  Repo: Reed et al. *Bootstrapping* (no official; common reproductions).

**Stage B — ELR** (Early-Learning Regularization) — `CE(p,y) − λ·log(1 − ⟨p, t⟩)`, with a
**per-sample EMA target** `t ← γ·t + (1−γ)·p`. Exploits early-learning: anchor to the belief the
net formed *before* it memorized noise; the regularizer cancels gradients on samples it already
disagrees with. Needs a per-row target buffer (stable sample indexing through the dataloader — the
only real plumbing) but no extra compute/network. Sweep λ∈{1,3,7}, γ=0.7. Repo:
`shengliu66/ELR` (official).

**Stage C — dual-network sample selection** (heavy; ~2× compute; gated — inherits E22's discount):

- **C1 · Co-teaching** — two nets; each epoch each keeps its own small-loss samples and teaches
  them to the *other* (disagreement stops mutual noise reinforcement). Repo: `bhanML/Co-teaching`.
- **C2 · JoCoR** — co-teaching + an agreement/co-regularization term so the two nets converge on a
  jointly-confident clean set. Repo: `hongxin001/JoCoR`.
- **C3 · DivideMix** — fit a 2-component GMM to per-sample loss → split clean/noisy, discard noisy
  labels, treat those rows as **unlabeled** in a MixMatch semi-supervised loop. SOTA on synthetic
  benchmarks, heaviest to implement/tune. Repo: `LiJunnan1992/DivideMix`.

## Gates (mirror E9 / E22)

- **Screen order (per the E22 negative prior):** run each arm on the **CE recipe first** — that's
  the only place E22 found noise-handling headroom (denoise helped CE +0.005–0.006, failed under
  LS). An arm that can't beat CE is dead. Only arms that clear CE go to the **LS-confirm** step.
- **Per arm:** report Δ vs **both** CE 0.7458 and LS 0.7565. "Beats CE but loses to LS" = a worse
  LS → close the arm. Promote to champion-confirm only if it **beats LS by ≥ +0.003** on the
  granite screen (the E9/E22 significance bar; slice noise ≈ ±0.003).
- **Stage B fires** only if the best Stage-A arm reaches **≥ LS − 0.002** (shows the robust-loss
  angle has life ELR could amplify); else ELR runs once as the strongest-single-net sanity check
  and the lane closes.
- **Stage C fires** only if A/B produces a **champion-confirmed** win — dual-net sample selection
  costs 2× compute and is a priori redundant with E22's denoising discount, so it must be earned.
- **Survivors:** confirm on the champion recipe (E8a+LS richargs `--full_data`) → qwen3 → LB.
  Granite screen ranks nothing final (E8 CV→LB reversal). Marginal champion-confirm win ⇒ build the
  submission zip per the recipe (no logit_bias; user uploads).

## Implementation plan (additive, backward-compatible — invariant)

New code defaults to OLD behavior; no existing flag renamed/repurposed; no running job affected.

- **Stage A:** extend `classification_loss()` + `--loss` choices in `src/finetune.py` with
  `gce/sce/apl/boot` and their hyperparams (`--gce_q`, `--sce_alpha/beta`, `--apl_alpha/beta`,
  `--boot_beta/mode`). Reuse the existing `make_loss_trainer` wiring. Smoke-test parity: `--loss ce`
  path stays byte-identical.
- **Stage B (ELR):** per-sample target buffer keyed by a stable row id threaded through
  `src/data.py` (add an index field, default unused); a `make_elr_trainer` that reads/updates the
  buffer in `compute_loss`.
- **Stage C:** a separate dual-network entry point (`experiments/noise-robust/co_teaching.py` style,
  like E22/E30 harvesters) — NOT folded into `finetune.py`. Only written if A/B unlocks the gate.
- Every entry point calls `log_cmd()`; every result records the verbatim command + log path + a
  figure (Δ-vs-anchor bars).

## Implementation status (✅ 2026-07-13 — all 8 arms coded + smoke-tested; nothing trained)

Additive & backward-compatible (invariant): `--loss ce` path byte-identical; new scripts standalone.
All entry points `log_cmd()`. **Smoke = plumbing only** (tiny data → macro-F1 ≈ 0; real learning needs
a box). Two-net arms (C1/C2/C3) OOM the local 8GB GPU / 15GB RAM → smoke on CPU + a `prajjwal1/bert-tiny`
stand-in; **train on a ≥16GB GPU**.

| Arm | Code | Faithful to | Smoke | Verified |
|---|---|---|:--:|---|
| A1–A4 | `src/finetune.py` `--loss gce/sce/apl/boot` (+`classification_loss`/`make_loss_trainer`/argparse) | HanxunH/APL · YisenWang/SCE · Reed'15 | ✅ | unit: finite grads all 4; GCE q→0≈CE(2.95 vs 3.01)/q=1≈MAE; full `finetune.py` runs `--loss gce` |
| B (ELR) | `experiments/noise-robust/elr.py` | shengliu66/ELR (`elr_loss.forward`) | ✅ 4060 | EMA target buffer + index-collate + reg + eval + CSV all run |
| C1 co-teach | `experiments/noise-robust/co_teaching.py` | bhanML/Co-teaching (`loss_coteaching`) | ✅ CPU | both nets, R(T) ramp, exchange, eval, CSV |
| C2 JoCoR | `experiments/noise-robust/jocor.py` | hongxin001/JoCoR (`loss_jocor`) | ✅ CPU | joint co-reg loss, single opt over both nets |
| C3 DivideMix | `experiments/noise-robust/dividemix.py` | LiJunnan1992/DivideMix (`eval_train`/`SemiLoss`/rampup) | ✅ tiny | warmup→GMM co-divide (per-net splits differ)→MixMatch→CSV |

**⚠️ DivideMix deviation (documented, per the official-code invariant):** the image `mixup(inputs)` has
no token-id analogue → we mix in **logit space**, which for a LINEAR final classifier is *exactly*
penultimate manifold mixup (`W(l·h_a+(1-l)·h_b)=l·logits_a+(1-l)·logits_b`); the two augmented views
become two **dropout** views. Everything else (GMM co-divide, co-refine/guess, sharpen, SemiLoss,
λ_u ramp, prior penalty, warmup) is a faithful port. A1–A4/B/C1/C2 are byte-faithful ports.

### Run commands (real screen — needs an assigned box; the box is the only remaining blocker)

Stage A (single-GPU, ≥12GB; standalone-vs-CE first per the E22-prior gate order):
```
for L in gce sce apl boot; do
  python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 \
    --loss $L --serialize v1 --max_len 512 --epochs 3 --batch_size 16 --lr 2e-5 \
    --out_dir output/e35/A_$L --tag e35_$L
done            # GCE sweep --gce_q {0.4,0.7,1.0}; SCE/APL tune --*_alpha/--*_beta; boot --boot_mode {soft,hard}
```
Stage B — ELR (single-GPU; +LS is the primary form):
```
python experiments/noise-robust/elr.py --ce_mode ls --elr_lambda 3 --elr_beta 0.7 \
  --epochs 5 --batch_size 32 --serialize v1 --max_len 512 --out output/e35/elr_ls.csv
```
Stage C — dual-net (≥16GB GPU; forget-rate sweep, low τ per E22 over-drop warning):
```
for FR in 0.06 0.10 0.20; do
  python experiments/noise-robust/co_teaching.py --forget_rate $FR --num_gradual 5 --epochs 5 \
    --batch_size 24 --grad_ckpt on --serialize v1 --max_len 512 --out output/e35/coteach_fr${FR}.csv
  python experiments/noise-robust/jocor.py --forget_rate $FR --co_lambda 0.1 --num_gradual 5 --epochs 5 \
    --batch_size 24 --grad_ckpt on --serialize v1 --max_len 512 --out output/e35/jocor_fr${FR}.csv
done
python experiments/noise-robust/dividemix.py --lambda_u 25 --p_threshold 0.5 --warmup 2 --epochs 6 \
  --batch_size 16 --grad_ckpt on --serialize v1 --max_len 512 --out output/e35/dividemix.csv
```
(run `-m src.finetune` from repo root; the Stage B/C scripts by path with `PYTHONPATH=.` — the dir hyphen
blocks `-m`. `--batch_size` tunes to the GPU — dual-net wants a decent batch for the small-loss selection.)

## Status

🟢 **CODE READY** (user 2026-07-13, "implement all") — all 8 arms coded + smoke-tested end-to-end;
**nothing trained** (local 8GB GPU / 15GB RAM can't hold the two-net arms). Awaits a ≥16GB box for the
real screen. Run order per gates: Stage A (CE-first) → B (if A shows life) → C1/C2/C3 (user un-gated C).
**Result:** —
