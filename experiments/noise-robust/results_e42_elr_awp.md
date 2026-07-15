# E42 · ELR + AWP combination (noise-robust loss × adversarial weights) — `research/noise-robust`

**Status: ✅ DONE (2026-07-14). 3 models, full_data, on NFS `output/e38/elr_awp/`.**
Combines [E35](results.md) ELR (early-learning regularization) with [E38](../optuna/results_e38.md)
AWP + its top optuna configs. Trained via `experiments/noise-robust/elr.py` (LS-CE + λ·ELR-reg +
AWP adversarial pass); on-disk tag `e38_*_elr`.

## Motivation

E38's AWP ensemble gained only **+0.0017** on the honest fold — every member shares the AWP+LS
recipe, so they share error structure (cf. E33 MoE, uniform-mean findings). **ELR is a different
loss** (an EMA-target regularizer that resists memorizing noisy labels) → different errors → the
error diversity the AWP-only pool lacked. E35 found ELR the sole additive winner over LS but
~noise-level at champion; the open question was whether ELR **stacks with AWP** on the tuned E38
configs and survives as a usable ensemble member. Prior worry: the standalone `elr.py` loop scored
~0.011 below the finetune Trainer on earlier fold runs.

## Method

- `elr.py`: base **LS-CE** (ε from the config) **+ λ·ELR-reg** (λ=3, β=0.7 — the E35 noise-robust
  winner) **+ AWP** (reusing `src.finetune.awp_perturb/awp_restore`), the config's AWP knobs.
- Applied to the **top-3 E38 optuna configs** (t070/t031/t040): each config's lr / ls / warmup / wd /
  awp_gamma / awp_lr, `--full_data`, **epochs=4**, richargs, bf16, 1/GPU on the sandbox 3090s.
- **Saves the last epoch** (ep4) — `elr.py` policy: the 3.5k full_data slice is too noisy for
  best-epoch selection, and epoch 4 empirically wins (E38 diagnostic + user).

## Results

Full_data, 3.5k held-out slice (last-epoch = ep4). **At parity with the AWP promotions — the feared
~0.011-below did NOT materialize** (tuned configs + full_data + ep4 held up):

| ELR+AWP model | slice F1 | AWP twin (E38) | Δ vs twin |
|---|---|---|---|
| t031_elr | 0.7856 | t031fd 0.7859 | −0.0003 (ties) |
| **t040_elr** | **0.7839** | t040fd 0.7814 | **+0.0025 (beats)** |
| t070_elr | 0.7836 | t070fd 0.7857 | −0.0021 |

vs base losses (elr.py CE 0.7458 / LS 0.7565): ELR+AWP = **+0.038 / +0.027**. Same strength as the
AWP twins, **different loss → genuine error diversity** — the intended use is as ensemble members on
top of the E38 AWP single (LB 0.79300), not as standalone singles.

### LB verdict — ELR adds nothing as a single (isolation test, 2026-07-14)

`submit_0714_elr_t031.zip` (t031_elr, fp16) submitted head-to-head against the **same-config** AWP
t031 (row 23):

| t031 config | 3.5k slice | **LB** |
|---|---|---|
| AWP (submit_0714_awp_t031) | 0.7859 | 0.79300 |
| **ELR+AWP (submit_0714_elr_t031)** | 0.7856 | **0.79311** |
| Δ (ELR effect) | −0.0003 | **+0.00011** |

**DEAD TIE** (LB noise ~1e-4). Slice (−0.0003) and LB (+0.00011) agree: **ELR is a no-op as a single
on top of AWP+LS** — confirms the E22/E35 prior *on the real leaderboard*, not just CV (LS already
absorbs the label noise; a second noise-handling mechanism has nothing left to do). The models are
not *identical* (they disagree on some rows), so ELR retains thin value as a same-config ensemble
ingredient, but the diversity lever with real headroom is **cross-backbone (qwen3, E41)**, not loss.
**Conclusion: don't spend further submissions on ELR as a standalone method.**

## Caveats

- ⚠️ **full_data** (66.5k train / 3.5k val) ⇒ trained on 75% of the 14k fold ⇒ **no `--session_fold`
  in elr.py ⇒ no fold twin ⇒ these cannot be honestly screened on the 14k.** ELR members' ensemble
  value is an **LB question**, not a fold-screen one (unlike the granite/qwen3 AWP fold models).
- 3.5k slice is noisy (±~0.001); the t031/t070 deltas vs their twins are within that band.

## Relation

Extends [E35 ELR](results.md) (loss) and [E38 AWP](../optuna/results_e38.md) (configs + twins).
Members feed the ensemble-buffer search above the E38 single-model SOTA (SUBMISSIONS.md row 23).
