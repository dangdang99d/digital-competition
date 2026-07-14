# E46 — inference-script upgrade screen (t031fd, 3.5k slice) + free ensemble checks

**Status: 🏃 RUNNING (2026-07-14, local ArchServer RTX 4060)** — user request: test the
inference-side candidates from the 2026-07-14 method search on the shared 3.5k held-out
slice against the `submit_0714_awp_t031.zip` model. Training-side siblings = **E47**
([results_e47.md](results_e47.md)).

## Setup

- **Model:** `output/e38/promote_top4/..._e38_t031fd` — the exact weights inside
  `submit_0714_awp_t031.zip` pre vocab-prune (parity-gated at zip build). Baseline slice
  self-F1 **0.7856** (pool cache `e38_t031fd.npz`); E46's own fp32 pass re-asserts parity
  (argmax agreement 1.0000, max |Δlogit| = 0).
- **Slice:** the shared 3.5k = 25% stratified hold-out of the seed-42 14k val
  (`_meta.npz`-asserted, identical to the ensemble pool slice). richargs, max_len 512.
- **Script:** `e46_infer_upgrades.py` (this dir) → `e46_infer_upgrades.{csv,log}` +
  per-arm predictions npz. Grids with label-picked hyperparams use honest 2-fold.
- ⚠️ **Read caveat:** the slice is CLEAN (no train→test shift), so adaptation arms
  (SHOT/SAR/T3A) are expected ≈0/negative here — E23 precedent: −0.0006…−0.0034 local
  → **+0.0019 LB**. Their slice job = correctness + collapse-safety + clean-data cost;
  LB judges the gain. MC-dropout / kNN / precision arms ARE honestly measurable here.
- ⚠️ t031fd trained with ALL dropout probs 0.0 → the MC-dropout arm is an **off-label
  probe** (injects dropout the model never trained with).

## Arms

| arm | what | config |
|---|---|---|
| base | fp32 forward, parity check | bs32, dataset order (= harvest protocol) |
| fp16 / fp16+fp32-head | ship-precision numerics; argmax flips vs fp32 | fresh per-precision loads (see bug note) |
| mcdrop | off-label MC-dropout, mean softmax | p ∈ {0.05, 0.10}, K=10 |
| shot | E23 SHOT/IM reference (entropy-min + IM diversity, encoder LN affines) | Adam lr ∈ {2e-4, 1e-3}, n=2048, bs8, 1 pass |
| sar | SAR-style (ICLR'23): entropy filter H<0.4·ln14 + SAM ρ=0.05 on LN affines | lr ∈ {2e-4, 1e-3}, same budget |
| t3a | T3A prototype classifier (NeurIPS'21), backprop-free, online | M ∈ {20, 50, 100, ∞} |
| knn | kNN label propagation over slice embeddings (instance-level smoothing, NOT per-class bias) | k ∈ {5,10,20,50} × α ∈ {.1,.2,.3} × steps ∈ {1,2}, **honest 2-fold pick** |

**Bug note (methodological, stays here per results-md invariant):** ModernBERT's compiled
embedding path caches the dtype of the FIRST forward — calling `.half()` on an
already-forwarded model then feeds fp32 hidden states into fp16 weights (crash). Fix:
`reference_compile=False` + a fresh `from_pretrained` per precision arm. Any future script
that re-casts a ModernBERT after a forward pass has this trap.

## Results — inference arms (2026-07-14, done)

⚠️ **On the 3.5k slice, which the user has since flagged NOISY** — treat every number here as
weak; the honest screen is the 14k-val E47. Adaptation arms are additionally clean-no-shift
(E23: local −0.003 → LB +0.002). base fp32 = 0.7856 (parity 1.0000).

| arm | slice mF1 | Δ vs base | flips | secs | note |
|---|---|---|---|---|---|
| base fp32 | 0.7856 | — | 0 | 74.8 | parity argmax 1.0000 |
| fp16 (ship precision) | 0.7858 | +0.0002 | 1 | 23.9 | **safe** — ship precision is score-neutral |
| fp16 body + fp32 head | 0.7856 | 0.0000 | 0 | 23.8 | bit-identical to fp32; no benefit |
| MC-dropout p=0.05 K=10 | 0.7857 | +0.0001 | 39 | 247 | OFF-LABEL (trained dropout 0.0) |
| **MC-dropout p=0.10 K=10** | **0.7867** | **+0.0011** | 63 | 249 | OFF-LABEL; mild gain even untrained-for → LB-probe candidate |
| SHOT lr=2e-4 | 0.7844 | −0.0012 | 9 | 172 | clean-slice neg (expected); gentle > hot |
| SHOT lr=1e-3 | 0.7825 | −0.0031 | 67 | 172 | overshoots (ent 1.41→0.31) |
| SAR lr=2e-4 | 0.7844 | −0.0012 | 6 | 271 | ≈ SHOT on clean slice; fewer flips (filter working) |
| SAR lr=1e-3 | 0.7811 | −0.0045 | 77 | 271 | overshoots |
| T3A M∈{20,50,100,∞} | 0.0053 | — | 3365 | <4 | ⚠️ **COLLAPSE to one class** — impl issue (prototype seeding), not viable as written |
| **kNN label-prop** (honest 2-fold) | **0.7872** | **+0.0016** | — | — | best arm; instance-level smoothing (allowed); honest-2-fold so less slice-overfit → LB-probe candidate |

**Reads:** (1) **fp16 is safe** to ship (neutral) — no need for the fp32-head split. (2)
**kNN smoothing (+0.0016) and MC-dropout p=0.1 (+0.0011)** are the only positive arms; both
are cheap inference-script changes and LB-probe candidates — but on a noisy slice, so low
confidence. (3) **SHOT/SAR clean-slice-negative** reproduces the E23 pattern (real gain only
under test-shift, LB judges); SAR≈SHOT here, no clear upgrade, gentle lr wins. (4) **T3A
collapsed** — my prototype-seeding impl drives all mass to one class; parked (would need the
official `matsuolab/T3A` filtered-support-set to trust). MC-dropout gains are OFF-LABEL (the
model trained with dropout 0.0) — a model trained WITH dropout (or an MSD/DropHead E47 winner)
would be the honest home for it.

## Results — free offline ensemble checks (averaging-space sweep, done 2026-07-14)

`experiments/ensemble/avg_space_sweep.py` on the 85-model 3.5k pool. Question: does a
different UNIFORM combining space beat shipped prob-mean? (Learned weights stay closed —
E29.) Note geometric-prob-mean ≡ logit-mean at argmax, so it is not a separate column.

| combo (3.5k slice) | prob_mean | logit_mean | rank_mean | prob_median | vote |
|---|---|---|---|---|---|
| SOTA trio t019+t023+t043 (LB 0.78780) | 0.7770 | 0.7774 | 0.7781 | **0.7790** | 0.7778 |
| SOTA trio + t031fd | 0.7815 | 0.7797 | **0.7817** | 0.7814 | 0.7809 |
| E38 quad (t001+t031+t040+t070 fd) | 0.7864 | 0.7854 | 0.7849 | 0.7856 | **0.7868** |
| t031fd+t070fd+t019 | **0.7860** | 0.7853 | 0.7859 | 0.7857 | 0.7856 |

- **Averaging space is within slice noise (±0.001–0.002, no consistent winner) — prob-mean
  keeps the ship slot.** No re-litigation unless an LB probe contradicts.
- **The real gap is member vintage:** every E38-member combo ≥0.781, all-E38 sets ≈0.786+
  vs the reigning trio's 0.7770 — the LB-SOTA ensemble (0.78780) predates every AWP/E38
  model. Greedy-forward (cap 4) reaches 0.789–0.791 on-slice (e.g. `t001fd + t031fd +
  t070_elr + e26s_ba`) but greedy-on-3.5k overfits and the slice mis-ranks ensembles
  (E8/E26) — candidates for an LB decision, not verdicts.
