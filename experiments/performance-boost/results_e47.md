# E47 — training-lever screen: structured label smoothing + head/gradient regularizers

**Status: 🟢 CODE READY (2026-07-14) — awaiting the 4×3090 instance the user will assign.**
Survivors of the 2026-07-14 method search (3-agent lit+competition sweep, cross-checked
against every closed negative) that are TRAINING-time levers. Inference-time levers are the
parallel screen **E46** ([results_e46.md](results_e46.md)); free offline ensemble checks
(averaging-space sweep) also live there. User-discarded before this screen: LLRD (user-tried,
no help), extra heads (user-tried, no help), truncation tricks (minimal effect, qwen3 barely
truncates), OOF soft-label blending (≡ E30 phase 1, α=0 won), fold-bagging (5× inference
cost), shared-init soup (deferred with E12v2 skepticism).

## Protocol (user 2026-07-14 — the honest 14k-val screen, NOT the noisy 3.5k slice)

The 3.5k held-out slice is **proven noisy** → not used here. Screen protocol:
1. **Train on 56k, eval on the full 14k val** — standard `split_indices(seed=42)`, NO
   `--full_data`. Per-epoch 14k-val macro-F1.
2. **Best epoch = the arm's score.** Run 6 epochs (AWP still climbing at ep3 — E43) to see
   the peak; read the per-epoch `eval_macro_f1` from the run log / `trainer_state.json`.
3. **Winner → retrain the SAME config from scratch on `--full_data` at the winning epoch →
   build zip → submit.** (See memory `screen-protocol-14k-not-3p5k`.)

**Anchor = the t031 recipe on THIS protocol (56k/14k), no new lever — RUN, not reused.** The
0.7856 slice / 0.79300 LB numbers are full_data, NOT comparable to a 14k-val screen; and the
E34 AWP figures (0.7804/0.7733) were the full_data 3.5k slice too → no in-protocol anchor
exists. Promote gate ≥ +0.003 vs the anchor's best-epoch 14k-val F1; member gate ≥ +0.001.

Anchor recipe command (drops `--full_data`, adds `--epochs 6`; t031 hyperparams verbatim):
```
python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 \
  --serialize richargs --loss ls --precision bf16 --init_seed 42 --group_by_length \
  --epochs 6 --max_len 512 --save_dtype fp16 --lr 3.466e-05 --batch_size 16 --grad_accum 1 \
  --warmup_ratio 0.1441 --label_smoothing 0.1365 --weight_decay 1.7158e-03 \
  --awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 \
  --out_dir output/e47 --results_name ft_e47.csv --tag e47_anchor
```

```
python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 \
  --serialize richargs --full_data --loss ls --precision bf16 --init_seed 42 \
  --group_by_length --epochs 4 --max_len 512 --save_dtype fp16 \
  --lr 3.466e-05 --batch_size 16 --grad_accum 1 --warmup_ratio 0.1441 \
  --label_smoothing 0.1365 --weight_decay 1.7158e-03 \
  --awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 \
  --out_dir output/e47 --results_name ft_e47.csv --tag e47_<arm>
```
(recovered verbatim from the t031fd promote log, `output/e38/vast_search/*.log` CMD line;
only out_dir/results_name/tag differ)

## Arms

Every arm = the anchor command ± ONE lever (AWP stays in — the deployment recipe).

| arm | lever | delta vs anchor command | rationale / source |
|---|---|---|---|
| anchor | none (t031 recipe, 56k/14k) | `--tag e47_anchor` | in-protocol 14k-val baseline (RUN) |
| A1 | **uniform NEGATIVE label smoothing** | `--loss lsmat --ls_matrix experiments/performance-boost/lsmat_neg01.npy` (GLS ε=−0.1; replaces the ε=0.1365 LS) | Wei et al. ICML'22 (2106.04149): positive-LS benefit reverses under high label noise (~our 20%) |
| A2 | **grouped ± label smoothing** (user 2026-07-14) | `--loss lsmat --ls_matrix experiments/performance-boost/lsmat_grouped_soft.npy` (in-group +0.05/class, out-group −0.005/class) | user idea: smooth INTO the confusion group (plausibly-true under ambiguity, E27: true ∈ top-1's group ≥95%), push out-group down. Groups from honest E30 OOF confusion (train-side only, no calibration): `{edit,write,apply_patch} {bash,tests,lint} {ask,plan,web_search} {read,grep,ls,glob} {respond_only}` — semantically clean. ⚠️ singleton group `respond_only` degenerates to pure negative-LS row |
| A3 | **DropHead** | `--drophead_p 0.1` | Zhou et al. 2020 (2004.13342), structured head dropout; ➖ prior: isotropic-noise family (R-Drop/NEFTune) all lost — this is the structured variant |
| A4 | **Child-Tuning-D** | `--child_p 0.3` | Xu et al. EMNLP'21 (2109.05687): fisher-masked gradients, +0.6–1.3 over FT tricks on GLUE; anti-memorization, composes with AWP |
| A5 | **multi-sample dropout head** | `--msd_k 5 --msd_p 0.3` | Inoue 2019 (1905.09788) + Jigsaw-8th writeup; head-only, ~free |

**Wave 1 (instance 44747112, 4×3090): anchor · A1 · A2 · A3.** Wave 2 (same box or the
incoming 2nd 4×3090): A4 · A5 + optional A1b ε=−0.05 / A2b `lsmat_grouped.npy` (+0.02/−0.01),
picked on wave-1 signs. Each run = 6 epochs, per-epoch 14k-val macro-F1 logged.

**New flags** (all additive, default-off, state_dict-compatible — hooks/monkeypatch, no module
renames): `--loss lsmat --ls_matrix` · `--drophead_p` · `--child_p --child_fisher_batches` ·
`--msd_k --msd_p`. Matrices built by `build_ls_matrix.py` (this dir; groups printed at build).
Smoke: `experiments/performance-boost/smoke_e47.sh` (runs each lever 1 step on `--limit 64`).

## Results

Score = best-epoch **14k-val** macro-F1 (56k/14k split, `split_indices seed=42`). Done
2026-07-14, vast 44747112 (anchor/negls/grpls/drophd) + 44587984 (child/msd), 6 epochs each.

Per-epoch 14k-val macro-F1:

| arm | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | **best** | Δ vs anchor | verdict |
|---|---|---|---|---|---|---|---|---|---|
| anchor (t031 AWP) | .6561 | .7532 | **.7782** | .7750 | .7747 | .7736 | **0.7782** @3 | — | baseline |
| A1 neg-LS ε=−0.1 | .6055 | .7041 | .7433 | .7615 | **.7783** | .7759 | 0.7783 @5 | **+0.0001** | ❌ flat |
| A2 grouped-LS | .5046 | .6824 | .7358 | .7589 | **.7767** | .7766 | 0.7767 @5 | **−0.0015** | ❌ hurts |
| **A3 DropHead p=0.1** | .6554 | .7468 | .7799 | .7795 | **.7826** | .7773 | **0.7826** @5 | **+0.0044** | ⚠️ screen-win but ❌ **LB 0.78866 < AWP 0.79300** (did NOT transfer — see below) |

⚠️ **LB VERDICT (2026-07-15): DropHead does NOT transfer.** `submit_0714_drophd.zip` (full_data
retrain, 4:55) → **LB 0.78866**, −0.0043 BELOW the AWP t031 single 0.79300. The +0.0044 14k-screen
win reversed. Cause = DropHead's stochastic masks inflate run-variance to ~0.005 (wave-2: 8-ep
p=0.1 0.7769 vs 6-ep 0.7826 = 0.0057 for the "same" config), so the single-run screen delta was
within noise; the 3.5k slice correctly predicted it (DropHead fd 0.7839 < AWP fd 0.7856). **AWP
t031 stays the SOTA single (0.79300).** E49 DropHead variants now low-value.
| A4 Child-Tuning-D 0.3 | .6608 | .7565 | **.7783** | .7741 | .7780 | .7754 | 0.7783 @3 | +0.0001 | ❌ flat |
| A5 MSD head K=5 | .6406 | .7650 | **.7792** | .7744 | .7753 | .7730 | 0.7792 @3 | +0.0010 | 🟡 marginal (member candidate) |

**Read-out:**
- **DropHead p=0.1 wins (+0.0044), best epoch 5.** The structured head-dropout beats the
  anchor AND delays the overfit peak from ep3→ep5 (regularizer effect). Notably it's the same
  stochastic-noise *family* as the failed R-Drop/NEFTune (E32) — but the **structured** (whole-
  head) variant lands where the isotropic ones didn't. → full_data retrain @ep5 → submit.
- **A5 MSD +0.0010** — sub-gate but positive and diverse (head-level mask ensemble); hold as
  an ensemble-member / DropHead-stack candidate (wave 2).
- **Both label-smoothing ideas fail on 14k-val:** neg-LS flat (+0.0001), grouped-LS −0.0015.
  The user's grouped ±smoothing and the GLS-negative paper both come out null-to-negative here
  — LS ε=0.1365 is already near-optimal; pushing wrong-class targets down doesn't help under
  AWP. Their curves also peak 2 epochs later (ep5) but at a lower ceiling → pure slowdown.
- **Child-Tuning flat** (+0.0001) — fisher-masking 70% of encoder grads neither helps nor hurts
  at this data scale (its GLUE gains were few-sample).
- Anchor's own 14k-val best-epoch **0.7782 @ep3** is the honest in-protocol baseline (distinct
  from the full_data slice 0.7856 / LB 0.79300).

## Wave-2 — DropHead p-sweep + MSD stacks (per-epoch 14k-val, 8 epochs)

⚠️ **These are 8-EPOCH runs — a DIFFERENT LR schedule from wave-1's 6-epoch (warmup+decay is
defined over total epochs), so NOT directly comparable to the 0.7826 wave-1 win.** Single-seed;
DropHead run-variance ~0.005, which exceeds the whole p-sweep spread → rankings unreliable.

| run | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | best |
|---|---|---|---|---|---|---|---|---|---|
| dh010_8ep (p=0.10) | .6155 | .7429 | .7734 | .7737 | **.7769** | .7725 | .7671 | .7615 | 0.7769 @5 |
| dh005 (p=0.05) | .6260 | .7373 | .7747 | .7712 | **.7777** | .7730 | .7672 | .7672 | 0.7777 @5 |
| dh015 (p=0.15) | .6121 | .7211 | .7725 | .7753 | **.7801** | .7740 | .7691 | .7653 | **0.7801 @5** |
| dh020 (p=0.20) | .6094 | .7110 | .7705 | .7746 | **.7770** | .7749 | .7692 | .7656 | 0.7770 @5 |
| dh025 (p=0.25) | .6038 | .7072 | .7558 | .7676 | **.7780** | .7747 | .7664 | .7664 | 0.7780 @5 |
| dh010 + MSD | .6226 | .7464 | .7721 | .7708 | **.7805** | .7736 | .7690 | .7645 | **0.7805 @5** |
| dh015 + MSD | .6222 | .7332 | .7681 | .7700 | .7759 | **.7763** | .7644 | .7627 | 0.7763 @6 |

Every run peaks at **ep5** then declines (ep6–8 overfit) → 8 epochs is strictly too long; the
peak is ep5 regardless of p. Best-of-8ep: p=0.15 (0.7801) and DropHead+MSD (0.7805).

## Schedule effect — DropHead p=0.1 at 6 vs 8 epochs (the confound)

| p=0.1 schedule | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | best |
|---|---|---|---|---|---|---|---|---|---|
| **6-epoch** (wave-1) | .6554 | .7468 | .7799 | .7795 | **.7826** | .7773 | — | — | **0.7826 @5** |
| **8-epoch** (wave-2) | .6155 | .7429 | .7734 | .7737 | **.7769** | .7725 | .7671 | .7615 | 0.7769 @5 |

The **8-epoch curve is uniformly BELOW the 6-epoch curve at every epoch** (the stretched LR
schedule leaves the model at a higher/less-decayed LR throughout) — peak −0.0057 for the "same"
lever. This is why the wave-2 / E49 8-epoch numbers can't be compared to the 6-epoch 0.7826, and
why epoch count must be PINNED (E38 fixed it at 4), not nudged run-to-run.

## DropHead vs the AWP king — 14k screen (clean: both 56k, 6-epoch, single-seed)

| model (14k screen) | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | best |
|---|---|---|---|---|---|---|---|
| AWP king (t031 recipe = anchor) | .6561 | .7532 | **.7782** | .7750 | .7747 | .7736 | 0.7782 @3 |
| DropHead p=0.1 | .6554 | .7468 | .7799 | .7795 | **.7826** | .7773 | **0.7826 @5** |

On the honest, schedule-matched 14k screen DropHead beat the AWP king by **+0.0044** (peaking 2
epochs later). This is the real evidence for a DropHead Optuna pass — but it's single-seed
(within ±0.005), and the full-data/LB reversal (0.78866 < 0.79300) says the delta didn't survive
uncontrolled epochs + 3.5k epoch-selection + noise. Full-data LB: AWP 0.79300 · DropHead 0.78866.

(fill: exact command + log path per run — invariant)
