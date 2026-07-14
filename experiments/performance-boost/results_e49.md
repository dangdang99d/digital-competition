# E49 — DropHead family: scheduling · MC-DropHead · structured-dropout siblings

⚠️ Renumbered E48→E49 (E48 taken by sequence-prior on the shared tree). Flag help in
`finetune.py` still reads "E49" now; no live artifacts predate the renumber.

**Status: ✅ DONE 2026-07-15** (8 arms, 5090 8×GPU, 8-epoch 14k screen). Best variant **S8
layer-ramp 0.7824** — nearly matches 6ep constant DropHead 0.7826 despite the depressed 8ep
schedule; but all within ±0.005 single-seed noise and the base DropHead already lost on LB
(0.78866 < AWP 0.79300, see [[e47-drophead-wins]]). Verdict: no arm is a shippable win; the
layer-ramp signal + the clean 14k DropHead>AWP delta motivate a **controlled Optuna** (pinned
epochs, multi-seed) rather than more single-run variants. M1/M2 not yet run. Follow-up to E47's DropHead win
(p=0.1 → **0.7826**, +0.0044 over the AWP anchor 0.7782 on 14k-val — the first structured
stochastic regularizer to beat the anchor where isotropic noise, R-Drop/NEFTune, failed).
E49 explores the family. Same protocol as E47 (memory `screen-on-14k-not-3p5k`): **14k-val
screen (56k/14k, per-epoch, best epoch), winner → full_data retrain → submit.** Anchor = the
same E47 anchor **0.7782 @ep3**; the DropHead p=0.1 constant baseline = **0.7826 @ep5**.

E47 wave-2 already runs the **p-sweep {0.05,0.15,0.20,0.25}** and **DropHead×MSD stacks** — E49
does NOT repeat those; it adds the axes wave-2 doesn't touch.

## Arms

**Training-time (14k-val screen, 8 epochs — DropHead peaks late; one run per single GPU):**

| arm | lever | flag | rationale |
|---|---|---|---|
| S1 | **Scheduled DropHead — updown** (paper's 0→p→0) | `--drophead_p <best> --drophead_schedule updown` | Zhou et al.'s actual method; constant-p (E47) was our deviation |
| S2 | **Scheduled DropHead — warmup** (0→p, then hold) | `--drophead_schedule warmup` | let the head-geometry settle before regularizing |
| S3 | **LayerDrop / stochastic depth** p=0.1 | `--layerdrop_p 0.1` | structured-DEPTH sibling; same "structured beats isotropic" logic that won DropHead (the earlier "marginal on 22 layers" dismissal assumed the isotropic framing) |
| S4 | **DropHead + LayerDrop** | `--drophead_p <best> --layerdrop_p 0.1` | do the two structured axes (head × depth) stack? |
| S5 | **DropFFN** p=0.1 block=64 | `--dropffn_p 0.1` | structured-WIDTH sibling (DropBlock on FFN intermediate neurons) |
| S6 | **token-wise DropHead** | `--drophead_p <best> --drophead_granularity token` | drop heads per-token instead of per-sequence (paper = sequence); finer noise |
| S7 | **correlated DropHead** | `--drophead_p <best> --drophead_correlated` | share ONE head mask across all layers → drop a head's full-DEPTH circuit (vs independent per layer) |
| S8 | **layer-ramp DropHead** | `--drophead_p <best> --drophead_layer_ramp` | scale p by depth (layer i uses p·(i+1)/L) — regularize task-specific top layers harder |

`<best>` = the wave-2 p-sweep winner (default 0.1 if the sweep is flat).

**Variant axes** (all on `install_drophead`, additive, default = E47 behavior): `--drophead_granularity
{sequence,token}` · `--drophead_correlated` · `--drophead_layer_ramp` (correlated ⊥ layer_ramp).
**Correctness tested** — `test_drophead.py`, 9/9 PASS on real granite: eval-invariance (hook
no-ops at inference), train/MC stochasticity, never-drop-all, drop-rate≈p, mask shapes,
MC variance-reduction (K=64 mean 10× more stable than single passes), cross-layer mask
sharing, layer-ramp monotonic (last 0.599 vs first 0.025 at p=0.6), mutual-exclusion guard.
⚠️ MC-DropHead is NOT an unbiased estimator of the clean pass (nH/kept rescaling + downstream
nonlinearities) — by design it's a different, ensembled prediction, not a denoiser.

**M1 — MC-DropHead (FREE, inference-only on a trained DropHead checkpoint):**
`mc_drophead_eval.py --k 16 --p {0.05,0.1,0.15}` — turn DropHead ON at inference, K passes,
avg softmax = free single-model ensemble; now **ON-LABEL** (E47 winner trained WITH head-
dropout, unlike E46's off-label MC-dropout +0.0011). Within the T4 budget for granite (K×5:10
headroom). Target = the wave-1 `e47_drophd` ckpt (0.7826, 56k-trained), scored on 14k val.

**M2 — SWA on the DropHead tail (needs a DEDICATED run — NOT free):** E43 killed SWA for "no
converged tail on 4-epoch runs"; DropHead pushes the peak to ep5 and we train 8 epochs, so the
tail now exists — BUT every DropHead run so far used `--keep_checkpoints 1` (best-snapshot only,
no per-epoch disk checkpoints). M2 therefore needs its own run: DropHead p=<best> 8ep
`--keep_checkpoints 5`, then `analysis/swa_average.py` over ep5–8. Queued as a training arm.

## Code (all additive, default-off, state_dict-compatible; ⚠️ single-GPU — break under DataParallel)

`src/finetune.py`: `--drophead_schedule {const,warmup,rampdown,updown} --drophead_warmup_frac`
(schedule callback mutates the shared DropHead state['p']) · `--layerdrop_p` (`install_layerdrop`,
residual-identity skip) · `--dropffn_p --dropffn_block` (`install_dropffn`, DropBlock on the FFN
down-proj input). `install_drophead` refactored to a shared `state={'p','infer'}` dict — `infer=True`
forces the mask active at eval (MC-DropHead). `experiments/performance-boost/mc_drophead_eval.py`
= M1. Smoke ✅ (updown+layerdrop+dropffn compose + train + eval on the 4060).

## Results

⚠️ **All E49 arms are 8-EPOCH** → the depressed-schedule regime (E47 §schedule: 8ep uniformly
below 6ep; matched baseline = 8ep p=0.1 DropHead **0.7769**, NOT the 6ep 0.7826). Single-seed,
±0.005 noise. ✅ **DONE 2026-07-15 (5090, all 8 arms 8 epochs). Every arm peaked at ep5; ep6–8
only overfit → best-epoch numbers below are FINAL.**

### Per-epoch 14k-val (best epoch = 5 for all; ep6–8 declined)

| arm | ep1 | ep2 | ep3 | ep4 | ep5 | best-so-far | Δ vs 8ep-DH 0.7769 |
|---|---|---|---|---|---|---|---|
| **S8 layer-ramp** | .5934 | .7449 | .7744 | .7763 | **.7824** | **0.7824** @5 | **+0.0055** |
| S1 updown | .6133 | .7409 | **.7794** | .7737 | .7779 | 0.7794 @3 | +0.0025 |
| S5 DropFFN | .6125 | .7422 | .7770 | .7724 | **.7791** | 0.7791 @5 | +0.0022 |
| S6 token-wise | .5869 | .7369 | .7726 | .7747 | **.7785** | 0.7785 @5 | +0.0016 |
| S2 warmup | .6074 | .7355 | .7727 | .7738 | **.7784** | 0.7784 @5 | +0.0015 |
| S7 correlated | .6348 | .7425 | **.7758** | .7749 | .7764 | 0.7764 @5 | −0.0005 |
| S3 LayerDrop | .5973 | .7392 | .7547 | .7683 | **.7742** | 0.7742 @5 | −0.0027 |
| S4 DropHead+LayerDrop | .5898 | .7191 | .7503 | .7669 | **.7723** | 0.7723 @5 | −0.0046 |

**Standout: S8 layer-ramp (0.7824) nearly matches the 6-epoch constant-p DropHead (0.7826)
despite the depressed 8-epoch schedule** — layer-ramp recovers the schedule penalty, the strongest
hint a better-than-p=0.1-constant config exists. S3/S4 (LayerDrop) hurt; correlated flat. All
within ±0.005 single-seed noise → directional only; motivates a *controlled* Optuna (pinned
epochs, multi-seed), not a verdict.

### Follow-ups
| arm | status |
|---|---|
| M1 MC-DropHead | 🔲 free — inference-only on the best DropHead ckpt |
| M2 SWA(DropHead ep5-8) | 🔲 needs dedicated `--keep_checkpoints 5` run |
