# E35 · AWP hyperparameter search (Optuna) — `research/optuna` (E28 harness, round 2)

**Status: 🏃 LAUNCHING (2026-07-13) — full search on rented 8×5090 (vast 44686633,
16 workers = 2/GPU, SQLite local). Search space + code cross-checked (ranges match
build_cmd suggest_* 1:1; anchor = enqueued LB-0.78557 config). Provisioning; CUDA/Blackwell
+ VRAM gates before launch.**

## Motivation

E34 AWP won at **blind defaults** (γ=1e-3, adv_lr=1e-4, start_epoch=1.0) → LB **0.78557**,
new single-model SOTA (beats E28 t043fd 0.78155 and every prior single). AWP is a
training-time lever with un-tuned knobs, so there is headroom. E35 tunes it with the E28
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
  --gpu_tag g<g> --n_trials 15 --exp e35 --search_awp --study e35_awp_granite
  --db output/optuna/e35.db --out_dir output/optuna/e35` (+ `--enqueue_anchor` on g0).

## Results

—
