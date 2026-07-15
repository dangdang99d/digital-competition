# E38 · AWP hyperparameter search (Optuna) — `research/optuna` (E28 harness, round 2)

> **Renumbered E35 → E38 (2026-07-13)** — the original E35 collided with the noise-robust
> experiment. The experiment number is **E38**; the LIVE run on vast keeps its `e35_*`
> runtime names (study `e35_awp_granite`, db `e35.db`, `--exp e35`, `output/optuna/e35/`,
> tags `e35_tNNN`) because it was launched before the renumber — harvest reads those paths.

**Status: 🏃 RUNNING (2026-07-13) — 16 workers live on rented 8×5090 (vast 44689856,
ssh1, 2/GPU SQLite-local). Gates passed: torch 2.7.1+cu128 runs on Blackwell sm_120
(bf16 matmul OK); VRAM 13.3GB/trial → 2/GPU packs in 31GB (~26GB, 0 OOM). Trials
generating (epochs=4 fixed, 8 params, NopPruner, anchor seeded). optuna installed into
the vast venv `/venv/main` (not in requirements.txt). Harvest top configs → `--full_data`
retrains → LB.**

## Motivation

E34 AWP won at **blind defaults** (γ=1e-3, adv_lr=1e-4, start_epoch=1.0) → LB **0.78557**,
new single-model SOTA (beats E28 t043fd 0.78155 and every prior single). AWP is a
training-time lever with un-tuned knobs, so there is headroom. E38 tunes it with the E28
Optuna machinery (which found t043 = +0.0138 on fold and promoted to real LB gains).

**Scope (user 2026-07-13): JOINT** — search the AWP knobs *and* the base recipe together,
anchored at champion. Rationale: AWP-on-champion (0.78557) already beats the E28-tuned
recipe alone (t043fd 0.78155), so the recipe optimum may *shift* once AWP is in the loss;
searching them jointly finds the recipe that best complements AWP rather than assuming the
champion recipe is still optimal under adversarial training.

## Search space

Base recipe (minus epochs) + AWP knobs, **8 searched params**:

| param | scale | range | champion/E34 anchor |
|---|---|---|:--:|
| `lr` | log | 5e-6 → 5e-5 | 2e-5 |
| `eff_batch` | categorical | {8,16,32} | 16 |
| `warmup_ratio` | uniform | 0.0 → 0.15 | 0.05 |
| `label_smoothing` | uniform | 0.02 → 0.20 | 0.10 |
| `weight_decay` | log | 1e-3 → 0.1 | 0.01 |
| **`awp_gamma`** | log | 5e-4 → 5e-3 | 1e-3 |
| **`awp_lr`** | log | 3e-5 → 3e-4 | 1e-4 |
| **`awp_start_epoch`** | categorical | {0, 1} | 1 |

- **`epochs` REMOVED from the search, FIXED at 4** (user 2026-07-13). We RAM-snapshot the
  BEST-epoch checkpoint, which makes "more epochs" a free lunch in the objective (more
  chances at a high best epoch) → a *searched* epochs drifts to 5 by noise (exactly what
  happened in E28: "epochs meaningless ≥3") without being genuinely better, and burns
  compute. Fixing it + best-epoch selection extracts each config's true peak. 4 contains
  the peak (E28 curves peak at ep3; AWP regularizes so its peak may sit a touch later, so
  4 > champion's 3 to avoid truncating it).
- **`awp_start_epoch` capped at {0,1}** (not {0,1,2}): AWP fires only from
  `state.epoch ≥ start_epoch`, so by epoch 2 BOTH values have AWP active — keeping
  MedianPruner meaningful. A start_epoch=2 trial would show 2 pre-AWP epochs that look
  like a plain run and mis-prune.
- **Anchor enqueued (trial 0):** the LB-validated config (champion recipe + AWP defaults =
  LB 0.78557) via `study.enqueue_trial`, so every trial reads as a Δ against a point whose
  LB value we actually know (E28's t024 trick).

## Objective / protocol

- **Objective:** session-grouped **fold-0** best-epoch macro-F1 (leak-free; E28 protocol).
  ⚠️ slice/fold mis-ranks (E8/E26) → **LB is the final judge** for any promotion.
- **Fixed:** granite-311m · richargs · `--loss ls` · `--init_seed 42` · max_len 512 ·
  bf16-auto · E19 fast shape (real batch, explicit `--grad_accum`, `--group_by_length`) ·
  **epochs = 4** (see search-space note: best-epoch snapshot would game a searched epochs).
- **Pruning: DISABLED (NopPruner)** (user 2026-07-13). Every trial runs its full 4 epochs.
  AWP's benefit appears only in the LATER epochs (activates from `start_epoch`), so
  early-epoch median-pruning would risk killing a slow-start-but-blooms config — and with
  epochs fixed at 4 on 16×5090, the compute pruning saves is marginal vs that downside.
  TPE random-startup bumped to 12 for the 8-param joint space. (E28 keeps MedianPruner.)
- **Promotion:** top-2–3 retrain from scratch via **`--full_data`** best-epoch (NOT
  `--all_data` — E28's last-epoch overfit trap that cratered t043 to LB 0.75934), then
  package + LB. Winners keep `--init_seed 42` (soup/ensemble-compatible).

## Implementation

`src/optuna_search.py` extended additively (default-off → E28 behavior byte-identical):
- `--search_awp` → `build_cmd` suggests the 3 AWP knobs and appends `--awp_gamma/--awp_lr/
  --awp_start_epoch`; `--exp e35` re-prefixes trial tags/dirs (no collision with E28's
  `e28_t*`); `--enqueue_anchor` (one worker only) seeds trial 0.
- AWP training path itself already smoke-validated (E34). Dry-run 2026-07-13 confirmed the
  joint command assembles correctly (recipe + AWP knobs, `e35_t000` tag).
- Launch (per worker): `CUDA_VISIBLE_DEVICES=<g> python -u -m src.optuna_search
  --gpu_tag g<g> --n_trials 200 --exp e35 --search_awp --study e35_awp_granite
  --db output/optuna/e35.db --out_dir output/optuna/e35` (+ `--enqueue_anchor` on g0).

## Results

**Search: 78 completed trials** (session-fold-0, 56k train / 14k holdout, epochs=4, NopPruner).
Best **t070 = 0.7806** vs the LB-validated anchor **t000 (AWP blind-defaults) = 0.7755** →
search gain **+0.0051** on fold. Top cluster t070/t031/t040/t054 all 0.7799–0.7806.

**Pattern read-out (what the search found):**
- **eff_batch 16 dominates** the top — every eff_batch=32 trial sank to the bottom tier
  (0.73–0.77); eff_batch 8 is mid. (Champion was 16.)
- **higher lr** than champion: top configs 2.3–4.3e-5 (champion 2e-5).
- **stronger AWP than the blind defaults**: awp_lr ≈2e-4 (≈2× the 1e-4 default), awp_gamma
  ≥1e-3 (up to 3.5e-3) across the top — the joint search's core finding: AWP wants a bigger
  adversarial step, and the recipe re-tunes hotter around it.
- **awp_start_epoch=1** in almost all top trials; **best-epoch = 4** (monotonic climb).
- label_smoothing spreads 0.05–0.20 (weak signal); weight_decay low-ish in the top.

**Epoch-extension diagnostic** (top-4 @ epochs=7, 56k fold): REFUTED "epoch-capped" — 7-epoch
runs peak e4–5 at ~0.777–0.778 and overfit after, never reaching the 4-epoch 0.7806. The
epoch-4 LR schedule was well-tuned. ⇒ **promote at epochs=4**. Curve:
[../../output/e38/epoch7_diagnostic_curve.md](../../output/e38/epoch7_diagnostic_curve.md).

**⚠️ Model weights:** the top-config 56k fold models (t070/t031/t040/t001) are now on NFS
`output/e38/vast_search/fold_models/` (5090 stopped after retrieval); t054's fold model was
pruned by the search (unavailable).
Full-data promotion of the top configs (epochs=4, best-epoch=4) → the submittable models.

### Honest ensemble screen — 56k-fold models on the leak-free 14k holdout

The 4 top-config **56k-fold** models (t070/t031/t040/t001, each held out the SAME session-fold-0
14k) screened by uniform softmax-prob mean on that holdout. Single-model F1s reproduced the
search values (t070 .7805 / t031 .7806 / t040 .7803 / t001 .7795) → the split byte-matches
`--session_fold 0`, so these numbers are honest. Script:
[e38_fold_screen.py](e38_fold_screen.py), cache `output/e38/fold_screen_probs.npz`.

| combo (uniform prob mean) | fold-0 14k F1 |
|---|---|
| **t031 + t040** (BEST) | **0.7822** |
| t070 + t031 + t040 + t001 (all-4) | 0.7820 |
| t070 + t031 + t001 | 0.7815 |
| t031 + t001 / t040 + t001 | 0.7814 |
| t070 + t040 + t001 | 0.7814 |
| t070 + t001 | 0.7812 |
| t031 + t040 + t001 | 0.7809 |
| t070 + t031 | 0.7808 |
| **best single (t031)** | **0.7806** |
| t070 + t040 | 0.7799 |
| t070 + t031 + t040 | 0.7797 |

**Read-out:** the AWP singles are strong (~0.7806, well above E28's fold singles ~0.772), but
the **ensemble gain is small — best is a PAIR (t031+t040) at +0.0017 over the best single**, and
adding t070/t001 *hurts*. Same-recipe AWP models share error structure (cf. E33 MoE / uniform-mean
findings) → little diversity to spend. Contrast E28 trio: fold 0.7774, +0.0055 gain → LB 0.78780.
**Projection** (E28 fold→LB offset ~+0.010): AWP pair 0.7822 → **~0.792 LB** (would clear the team's
0.78916; near #12 = 0.79282). ⇒ **ship t031fd+t040fd pair (primary); t031fd single (fallback).**

### Training-data provenance (which models are full_data vs 56k fold)

- **All 78 search trials `tNNN` below → trained on the 56k `--session_fold 0` split** (56k
  train / 14k holdout). The "fold F1" column is the honest 14k-holdout score. NONE are
  full_data. Weights live on the parked 5090 (`output/optuna/e35/ft_*_e35_tNNN`).
- **`--full_data` models (66.5k train / 3.5k holdout)** — trained SEPARATELY, not in the
  table below:
  - **Promotions `tNNNfd`** (t070fd · t040fd · t031fd · **t001fd**) — the top configs
    retrained on full_data, epochs=4, best-epoch=4 (t054 swapped for t001 — t054 had no
    surviving 56k-fold model to screen against). Submittable models (sandbox,
    `output/e38/promote_top4/`).
  - **E34 `e34_c_awp`** (the shipped `submit_0713_awp.zip`, LB 0.78557) — champion recipe +
    AWP blind-defaults, **epochs=3**, full_data. ⚠️ full_data ⇒ cannot be honestly evaluated
    on the 14k holdout (trained on 75% of it).
- **⚠️ honest-OOF note:** any ensemble screen on the 14k holdout must use the **56k-fold
  `tNNN` weights** (leak-free), NOT the full_data `*fd` / E34 models.

### Full-data promotion results (per-epoch 3.5k slice; shipped = load_best epoch)

Top configs retrained on full_data (66.5k train / 3.5k val), epochs=4, `--keep_checkpoints 2`.
`load_best_model_at_end` picked the best-of-4 on the 3.5k slice per model. ⚠️ 3.5k is noisy
(±~0.001) — fair for single-vs-single, NOT the leak-free 14k, and these cannot be fold-screened.

| model | ep1 | ep2 | ep3 | ep4 | shipped | slice F1 |
|---|---|---|---|---|---|---|
| t070fd | .6812 | .7723 | .7794 | **.7857** | ep4 | **0.7857** |
| t031fd | .6741 | .7759 | .7817 | **.7859** | ep4 | **0.7859** |
| t040fd | .7048 | .7647 | **.7814** | .7798 | ep3 | **0.7814** |
| t001fd | — | — | — | — | best | **0.7858** |

**vs the shipped E34 AWP (0.7804 slice → LB 0.78557):** all four beat it by +0.0010–0.0055 on the
identical slice (t031fd best single, **+0.0055**). Confirms the joint search + epochs 4 stacked.
Models: `output/e38/promote_top4/`. (t040 keeps ep3 = its own slice best per user; the ep4-forcing
was dropped since the noisy slice genuinely preferred ep3 there by +0.0016.)

### ELR + AWP diversity members → **E42** (separate experiment)

The ELR × AWP combination on the top-3 E38 configs (t031_elr 0.7856 / t040_elr 0.7839 / t070_elr
0.7836 — at parity with the AWP twins, cross-loss diversity members) is its own experiment:
**[E42](../noise-robust/results_e42_elr_awp.md)** (uses E38's configs + twins). Models:
`output/e38/elr_awp/`.

### All 78 trials — **56k session-fold-0** trained (params + honest 14k-fold F1, sorted)

**⭐ = selected top-4 for `--full_data` promotion on the 3090s** — aligned to the 4 configs with retrievable 56k-fold models (t054's fold weights were pruned by the search, so it was swapped for t001 to keep the honest ensemble screen coherent) →
`t070fd / t040fd / t031fd / t001fd`, epochs=4, best-epoch=4 (these become the submittable models).

| tag | fold F1 | best ep | lr | eff_batch | warmup | label_smooth | weight_decay | awp_gamma | awp_lr | awp_start |
|-----|:---:|:---:|---|:---:|---|---|---|---|---|:---:|
| **t070 ⭐** | 0.7806 | 4 | 2.932e-05 | 16 | 0.1462 | 0.0518 | 3.20e-03 | 3.421e-03 | 2.063e-04 | 1 |
| **t031 ⭐** | 0.7805 | 4 | 3.466e-05 | 16 | 0.1441 | 0.1365 | 1.72e-03 | 2.012e-03 | 2.309e-04 | 1 |
| **t040 ⭐** | 0.7805 | 4 | 2.291e-05 | 16 | 0.1264 | 0.1713 | 1.58e-03 | 1.140e-03 | 2.117e-04 | 1 |
| t054 | 0.7799 | 4 | 4.254e-05 | 16 | 0.1412 | 0.0728 | 1.72e-03 | 3.459e-03 | 1.719e-04 | 1 |
| **t001 ⭐** | 0.7792 | 3 | 3.273e-05 | 8 | 0.0336 | 0.1686 | 6.53e-03 | 7.237e-04 | 2.848e-04 | 1 |
| t029 | 0.7792 | 4 | 3.760e-05 | 16 | 0.1335 | 0.1989 | 1.70e-02 | 2.290e-03 | 2.689e-04 | 1 |
| t044 | 0.7790 | 4 | 2.840e-05 | 8 | 0.0108 | 0.1314 | 1.21e-02 | 8.308e-04 | 2.268e-04 | 1 |
| t022 | 0.7789 | 4 | 2.868e-05 | 8 | 0.0789 | 0.1586 | 2.98e-02 | 9.075e-04 | 2.567e-04 | 0 |
| t042 | 0.7789 | 3 | 2.710e-05 | 8 | 0.0207 | 0.1865 | 4.28e-03 | 8.293e-04 | 1.333e-04 | 1 |
| t046 | 0.7788 | 3 | 3.016e-05 | 8 | 0.0000 | 0.1997 | 4.36e-03 | 7.837e-04 | 1.878e-04 | 1 |
| t061 | 0.7787 | 4 | 4.879e-05 | 16 | 0.1174 | 0.0503 | 5.03e-03 | 2.500e-03 | 1.491e-04 | 0 |
| t032 | 0.7785 | 4 | 2.674e-05 | 16 | 0.0866 | 0.1933 | 6.05e-03 | 1.664e-03 | 2.396e-04 | 1 |
| t034 | 0.7785 | 4 | 4.014e-05 | 8 | 0.0041 | 0.1831 | 3.24e-03 | 1.235e-03 | 2.628e-04 | 1 |
| t043 | 0.7781 | 3 | 4.107e-05 | 8 | 0.0331 | 0.1829 | 2.47e-02 | 9.565e-04 | 2.039e-04 | 1 |
| t067 | 0.7781 | 4 | 2.389e-05 | 16 | 0.1348 | 0.1996 | 1.16e-03 | 6.331e-04 | 1.491e-04 | 0 |
| t051 | 0.7781 | 4 | 2.034e-05 | 8 | 0.0596 | 0.1795 | 1.58e-02 | 7.794e-04 | 2.290e-04 | 0 |
| t027 | 0.7781 | 4 | 1.731e-05 | 16 | 0.1129 | 0.1689 | 1.12e-03 | 1.453e-03 | 2.506e-04 | 1 |
| t035 | 0.7779 | 4 | 2.891e-05 | 16 | 0.1161 | 0.1895 | 5.02e-03 | 2.334e-03 | 1.650e-04 | 1 |
| t065 | 0.7778 | 4 | 4.796e-05 | 16 | 0.1429 | 0.1831 | 3.74e-02 | 2.526e-03 | 2.466e-04 | 1 |
| t038 | 0.7778 | 4 | 3.112e-05 | 16 | 0.0930 | 0.1785 | 3.38e-03 | 1.548e-03 | 1.749e-04 | 1 |
| t068 | 0.7777 | 4 | 2.921e-05 | 16 | 0.1400 | 0.1385 | 2.58e-02 | 3.543e-03 | 2.701e-04 | 1 |
| t037 | 0.7777 | 4 | 2.975e-05 | 16 | 0.1143 | 0.1814 | 5.33e-03 | 1.680e-03 | 2.850e-04 | 1 |
| t048 | 0.7776 | 4 | 1.885e-05 | 8 | 0.0158 | 0.1565 | 2.59e-03 | 9.175e-04 | 2.459e-04 | 1 |
| t030 | 0.7774 | 4 | 2.863e-05 | 16 | 0.0807 | 0.1711 | 3.21e-03 | 2.557e-03 | 2.277e-04 | 1 |
| t063 | 0.7773 | 4 | 4.164e-05 | 16 | 0.1064 | 0.1949 | 2.99e-02 | 2.165e-03 | 2.776e-04 | 0 |
| t047 | 0.7769 | 3 | 3.374e-05 | 8 | 0.0555 | 0.1859 | 2.90e-02 | 8.164e-04 | 1.934e-04 | 1 |
| t008 | 0.7767 | 4 | 3.085e-05 | 16 | 0.0968 | 0.1676 | 3.84e-03 | 1.687e-03 | 2.367e-04 | 1 |
| t069 | 0.7767 | 4 | 3.667e-05 | 16 | 0.1420 | 0.0300 | 1.95e-03 | 2.876e-03 | 1.747e-04 | 1 |
| t033 | 0.7764 | 4 | 3.367e-05 | 8 | 0.0480 | 0.1992 | 7.21e-03 | 1.440e-03 | 1.971e-04 | 1 |
| t052 | 0.7762 | 3 | 3.072e-05 | 8 | 0.0011 | 0.1789 | 1.16e-02 | 7.088e-04 | 2.277e-04 | 1 |
| t055 | 0.7762 | 4 | 3.213e-05 | 16 | 0.1497 | 0.1205 | 1.28e-03 | 2.505e-03 | 2.258e-04 | 1 |
| t028 | 0.7761 | 3 | 2.676e-05 | 16 | 0.0720 | 0.1369 | 7.74e-03 | 1.773e-03 | 1.108e-04 | 1 |
| t064 | 0.7761 | 4 | 4.173e-05 | 16 | 0.1342 | 0.0394 | 3.69e-03 | 2.853e-03 | 1.470e-04 | 0 |
| t049 | 0.7760 | 3 | 2.888e-05 | 8 | 0.0093 | 0.1534 | 3.90e-03 | 7.377e-04 | 2.451e-04 | 1 |
| t059 | 0.7759 | 4 | 4.242e-05 | 16 | 0.1300 | 0.0219 | 1.26e-03 | 2.522e-03 | 1.648e-04 | 1 |
| t056 | 0.7759 | 3 | 3.232e-05 | 8 | 0.1284 | 0.0822 | 2.36e-03 | 1.636e-03 | 1.497e-04 | 1 |
| t058 | 0.7757 | 4 | 2.659e-05 | 16 | 0.1440 | 0.1063 | 1.69e-03 | 3.425e-03 | 2.617e-04 | 1 |
| t041 | 0.7757 | 3 | 4.346e-05 | 8 | 0.0351 | 0.1796 | 1.06e-02 | 6.920e-04 | 2.554e-04 | 1 |
| t000 | 0.7755 | 4 | 2.000e-05 | 16 | 0.0500 | 0.1000 | 1.00e-02 | 1.000e-03 | 1.000e-04 | 1 |
| t007 | 0.7755 | 4 | 1.800e-05 | 16 | 0.0609 | 0.1030 | 2.71e-03 | 5.560e-04 | 2.266e-04 | 1 |
| t045 | 0.7755 | 4 | 3.813e-05 | 8 | 0.0416 | 0.1180 | 6.54e-03 | 1.103e-03 | 2.507e-04 | 1 |
| t039 | 0.7754 | 3 | 3.236e-05 | 8 | 0.0919 | 0.1687 | 9.32e-03 | 8.375e-04 | 2.872e-04 | 1 |
| t036 | 0.7753 | 3 | 3.415e-05 | 8 | 0.0837 | 0.1611 | 2.56e-02 | 6.480e-04 | 2.709e-04 | 1 |
| t053 | 0.7753 | 3 | 4.209e-05 | 8 | 0.0545 | 0.1767 | 6.07e-03 | 5.230e-04 | 2.466e-04 | 1 |
| t060 | 0.7749 | 3 | 4.132e-05 | 16 | 0.1487 | 0.0379 | 1.32e-03 | 3.900e-03 | 8.035e-05 | 1 |
| t071 | 0.7749 | 3 | 4.129e-05 | 16 | 0.1236 | 0.0217 | 1.73e-03 | 2.233e-03 | 2.035e-04 | 1 |
| t072 | 0.7745 | 3 | 4.917e-05 | 16 | 0.1288 | 0.1351 | 3.70e-02 | 1.624e-03 | 2.962e-04 | 1 |
| t019 | 0.7745 | 4 | 8.609e-06 | 8 | 0.1221 | 0.1515 | 1.42e-03 | 4.397e-03 | 4.084e-05 | 0 |
| t018 | 0.7742 | 3 | 2.964e-05 | 32 | 0.0200 | 0.1223 | 1.14e-03 | 6.077e-04 | 1.831e-04 | 0 |
| t050 | 0.7741 | 3 | 3.464e-05 | 8 | 0.0361 | 0.1903 | 4.86e-03 | 5.648e-04 | 2.103e-04 | 1 |
| t015 | 0.7733 | 4 | 1.237e-05 | 8 | 0.0954 | 0.0487 | 9.92e-03 | 7.673e-04 | 4.534e-05 | 0 |
| t062 | 0.7730 | 3 | 4.188e-05 | 8 | 0.1137 | 0.0898 | 4.04e-03 | 3.278e-03 | 1.680e-04 | 1 |
| t004 | 0.7729 | 4 | 9.054e-06 | 8 | 0.0194 | 0.0510 | 2.47e-03 | 8.378e-04 | 6.119e-05 | 0 |
| t009 | 0.7724 | 3 | 3.955e-05 | 8 | 0.0606 | 0.1379 | 1.21e-03 | 3.223e-03 | 4.668e-05 | 1 |
| t003 | 0.7708 | 4 | 1.303e-05 | 32 | 0.1422 | 0.1806 | 4.32e-02 | 1.412e-03 | 1.685e-04 | 0 |
| t057 | 0.7707 | 4 | 2.959e-05 | 32 | 0.1305 | 0.1244 | 1.08e-03 | 1.950e-03 | 1.473e-04 | 1 |
| t006 | 0.7699 | 4 | 1.081e-05 | 16 | 0.1192 | 0.0287 | 3.05e-02 | 5.961e-04 | 1.489e-04 | 1 |
| t014 | 0.7692 | 4 | 6.237e-06 | 8 | 0.0089 | 0.1699 | 2.00e-03 | 7.909e-04 | 8.037e-05 | 0 |
| t023 | 0.7687 | 3 | 4.254e-05 | 32 | 0.0707 | 0.0251 | 4.05e-02 | 1.954e-03 | 4.360e-05 | 0 |
| t020 | 0.7681 | 4 | 1.386e-05 | 32 | 0.0706 | 0.1396 | 1.34e-03 | 7.719e-04 | 1.379e-04 | 1 |
| t016 | 0.7667 | 4 | 1.575e-05 | 32 | 0.0908 | 0.0869 | 7.10e-02 | 1.594e-03 | 4.544e-05 | 0 |
| t010 | 0.7666 | 3 | 2.215e-05 | 32 | 0.0465 | 0.1127 | 2.39e-03 | 1.952e-03 | 4.867e-05 | 1 |
| t024 | 0.7665 | 4 | 1.403e-05 | 32 | 0.1288 | 0.1525 | 5.51e-02 | 2.942e-03 | 1.917e-04 | 1 |
| t012 | 0.7656 | 4 | 1.206e-05 | 32 | 0.0543 | 0.0577 | 1.73e-03 | 7.888e-04 | 4.318e-05 | 0 |
| t002 | 0.7648 | 3 | 2.501e-05 | 32 | 0.0059 | 0.0222 | 2.89e-03 | 6.575e-04 | 5.278e-05 | 1 |
| t073 | 0.7645 | 2 | 4.613e-05 | 8 | 0.1203 | 0.1586 | 9.02e-02 | 2.699e-03 | 2.454e-04 | 1 |
| t025 | 0.7642 | 4 | 1.068e-05 | 32 | 0.0414 | 0.0667 | 1.89e-02 | 3.752e-03 | 1.148e-04 | 1 |
| t066 | 0.7604 | 2 | 4.134e-05 | 8 | 0.1305 | 0.1823 | 5.26e-02 | 3.024e-03 | 2.062e-04 | 1 |
| t026 | 0.7582 | 3 | 5.953e-06 | 16 | 0.0746 | 0.0702 | 5.61e-03 | 9.696e-04 | 4.100e-05 | 1 |
| t021 | 0.7576 | 4 | 5.337e-06 | 8 | 0.1125 | 0.1623 | 2.19e-02 | 6.035e-04 | 1.948e-04 | 0 |
| t011 | 0.7521 | 4 | 7.494e-06 | 32 | 0.0948 | 0.1137 | 9.73e-02 | 1.806e-03 | 4.816e-05 | 0 |
| t017 | 0.7446 | 4 | 6.629e-06 | 32 | 0.0165 | 0.1391 | 1.84e-02 | 8.841e-04 | 7.766e-05 | 1 |
| t005 | 0.7363 | 4 | 5.126e-06 | 32 | 0.0679 | 0.1306 | 1.72e-02 | 5.136e-04 | 1.935e-04 | 0 |
| t013 | 0.7314 | 3 | 5.172e-06 | 32 | 0.1309 | 0.0722 | 1.28e-02 | 3.764e-03 | 1.359e-04 | 0 |
| t074 | 0.7211 | 1 | 4.677e-05 | 16 | 0.1245 | 0.1599 | 4.76e-02 | 2.510e-03 | 2.333e-04 | 0 |
| t075 | 0.7180 | 1 | 4.686e-05 | 8 | 0.1280 | 0.1856 | 4.24e-02 | 2.584e-03 | 1.535e-04 | 1 |
| t079 | 0.6827 | 1 | 3.088e-05 | 32 | 0.0362 | 0.1117 | 3.48e-02 | 7.833e-04 | 2.357e-04 | 1 |
| t076 | 0.6384 | 1 | 4.102e-05 | 32 | 0.0647 | 0.1218 | 7.76e-02 | 9.317e-04 | 2.346e-04 | 0 |

---
_Ops/launch gotchas for multi-worker vast runs (thread caps, venv python, dud hosts, etc.):
[vast_multiworker_runbook.md](vast_multiworker_runbook.md)._
