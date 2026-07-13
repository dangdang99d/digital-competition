# E33 · Model-level MoE — jointly trained experts + controller — `research/moe`

**Status: 🔲 QUEUED (user 2026-07-13). No runs yet.**

## Motivation — the one combination mechanism never tried

Every combination result so far is **post-hoc over independently trained, frozen models**:

| mechanism | result |
|---|---|
| uniform prob mean | 🏆 trio LB **0.78719** — the reigning SOTA |
| global weights (w424) | LB 0.78709 ≈ uniform — no help |
| fitted stackers (E29 ①) | ≤ uniform on honest 70k OOF |
| per-row gating over frozen members (E29 ③/④) | −0.004…−0.009; NLL gate provably collapses onto the best-calibrated member |
| distillation (E26-1B, E30) | closed twice — trio edge = inference-time averaging, not distillable |

What has **never** been tested is **joint training**: experts and a router optimized
together, so experts can specialize *because* the router exists (co-adaptation). E29's
gates could only reweight fixed opinions; a jointly trained gate shapes what the experts
become. That is the untested hypothesis.

**Priors are honestly against it** — E1 (hand-partitioned specialists −0.11 vs
generalist), E29 ④ (trained gates trade away error diversity), and the trio's edge being
plain averaging of diverse errors. One cheap joint-training screen either closes the
question or hands E26 a new kind of member.

## Architecture (PINNED 2026-07-13: shared-trunk, pre-expert routing — user accepted)

User-picked shape: **1 controller · 4 experts · soft gate during training.** Controller
construction settled 2026-07-13 after design discussion:

- **Shared embedding layer** (single vocab-pruned granite embedding matrix — emb is 64.6%
  of granite params; sharing it is what makes 4 experts packageable, see Packaging).
- **Shared trunk = the controller's feature extractor:** granite layers **1–k, k=6**
  (~27% of the 22-layer stack), shared by all experts. The "cutoff granite" IS the
  trunk — not a separate parallel model.
- **1 controller (gate net):** mask-aware mean-pool of the trunk's hidden states,
  concatenated with free structure scalars (log1p history-event count · first-step flag),
  LayerNorm → small MLP → softmax over 4 experts. KBs of params. Scalars come from the
  raw record, not re-parsed text; they also make gate histograms interpretable by regime.
- **4 experts** = granite layers k+1…22 (each expert its own copy), each with its own
  14-class head → per-expert softmax probs.
- **Blend:** gate-weighted mean of expert probs (same mechanics as the trio's uniform
  mean, weights learned per row).
- **k is a real knob:** too high strangles expert diversity, k=0 degenerates to a
  bag-of-embeddings gate. k=6 for stage 1; sweep k ∈ {3, 6, 9} only in stage 2 on signal.

**Why pre-expert routing (and not the alternatives):** canonical MoE routes on the
input-side representation — Jacobs et al. 1991 defines g(x); Switch/GShard/Mixtral route
per-layer on the hidden state ENTERING the MoE layer (= shared trunk by construction);
shared-bottom/MMoE is the same skeleton in multi-task learning. Rejected for stage 1:
- **Separate cutoff-granite controller** — duplicates params AND a forward pass the
  experts' own bottom layers already do; worsens the time budget for nothing.
- **Expert-features → controller** (experts run first, pooled features feed the gate) —
  structurally E29 arm ④, the shape that collapsed onto the best-calibrated member;
  forfeits top-k routing (must run ALL experts → no time lever); rich-get-richer
  feedback (gate rewards confident experts → they get more gradient → more confident);
  and gate input becomes "who looks confident," which E20/E27 showed saturates at a
  ~0.85 ceiling reflecting intrinsic ambiguity, not competence. Kept as the named
  **stage-2 gate-input variant** — it must earn its run by stage 1 showing the gate is
  signal-starved (near-uniform weights everywhere despite healthy experts).
- **Loss:** LS-CE (ε=0.1) on the blended distribution **+ load-balancing auxiliary**
  (Switch-Transformer-style importance loss; official-code invariant → diff our aux-loss
  against the HF `transformers` Mixtral/Switch `load_balancing_loss_func`, note version)
  **+ gate-entropy floor** — both anti-collapse guards are non-optional (E29 ④ showed
  exactly this failure: gate drifts to one member).
- **Optional diversity lever (stage-2 knob, not stage 1):** expert dropout — randomly
  mask experts per batch and renormalize gate weights.
- **Init:** all 4 expert ENCODER stacks byte-identical pretrained copies; each expert's
  classifier Linear freshly initialized with a DISTINCT derived seed (`init_seed*1000+e`).
  Implementation finding (2026-07-13): granite's config has **all dropouts = 0.0**, so
  4 byte-identical experts under a fresh near-uniform gate would receive near-identical
  gradients and barely diverge — distinct classifier init is the standard, minimal
  symmetry breaker. E12's shared-init rule is NOT violated in spirit: it forbids
  weight-averaging across inits (soups); nothing in the MoE is ever weight-averaged,
  and unbroken symmetry is itself the degenerate case here.

**⛔ PARKED — FFN-level sparse MoE (upcycling):** user unsure (2026-07-13). Also
disfavored on merits: FFN = only 18.7% of granite params (small capacity gain) and
upcycling = warm-start of the champion, the E24 recovery-trap regime. Revisit only on
explicit user re-spec.

## Protocol

- **Recipe:** champion E8a+LS lineage (richargs + full_data + bf16), **from scratch**
  (E24: never warm-start), one joint training. Exactly one new mechanism vs champion:
  the 4-expert + gate architecture.
- **Baselines:**
  - single-model anchor: champion full_data CV **0.7803**;
  - post-hoc-averaging control (the scientific null): **uniform mean of the SAME 4
    experts** evaluated post-training with the gate bypassed — free at eval time, isolates
    "does the learned gate beat averaging its own experts";
  - context: trio honest-OOF uniform null 0.7745 / LB 0.78719.
- **Read-outs (all uncal macro-F1, same eval slice):** blended MoE vs 0.7803 · blended
  MoE vs its own uniform-expert mean · per-expert solo scores + gate weight histogram
  (collapse diagnostic: max mean gate weight, per-row entropy) · first-step slice.
- **Gates:** package at ≥ +0.003 vs 0.7803 full_data CV. Marginal +0.001–0.003 → treat
  as ensemble-member/diverse-regime candidate, judged by honest OOF / LB (E8/E26: slice
  mis-ranks — LB judges any final claim). Gate collapsed onto one expert (max mean weight
  > ~0.9) with blend ≈ best expert → negative result, write it up, close.
- **Stage 2 (only if stage 1 clears or near-misses):** expert-dropout knob · trunk-depth
  sweep k ∈ {3, 6, 9} · gate-input variant (expert-features → gate; only if stage-1 gate
  is signal-starved) · top-k inference probe (soft-trained gate, top-2 at eval — time
  lever, see Packaging).

## Packaging / constraints (why shared embedding is load-bearing)

- **Zip ≤1GB:** 4 independent granites are DEAD by E26's own math (3 fp16 vocab-pruned =
  836M; a 4th breaks the cap). Shared-trunk ships ONE emb table + 6 shared layers +
  4×16 expert layers + KB gate ≈ 70/88 of the layer-copies of four full stacks, on top
  of the embedding savings — estimated well under 1GB fp16; exact size measured at
  packaging, parity gate as always.
- **T4 ≤10min:** shared trunk pays for itself — one pass = (6 + 4×16)/22 ≈ **3.2
  full-pass-equivalents ≈ ~7:15 projected** (vs ~9:10 for 4 independent granites), and
  the gate fires at the fork so **top-2 routing ≈ (6 + 2×16)/22 ≈ 1.7 passes ≈ ~4:00**.
  Anchors-only rule applies: nearest anchor = trio 6:52 (3 full passes); these are
  like-for-like layer-count scalings of it, estimates not guarantees. First submission =
  most conservative variant (bank the anchor).
- Ship shape: vocab-prune + fp16 + parity gate; NO logit calibration (gate reweights
  member opinions — same exemption class as the trio's uniform mean; flag to user with
  the zip regardless, per the E29 precedent of asking).

## Implementation (✅ built + smoke-tested 2026-07-13 — nothing launched)

- **`src/moe_model.py`** — `GraniteMoE`: splits one loaded
  `ModernBertForSequenceClassification` into shared embeddings + trunk (layers 0..k-1,
  `final_norm` moved into the branches) + gate + N `ExpertBranch`es (layers k..21 +
  own final_norm/head/classifier). Embeddings and the 4D/sliding-window masks are
  computed ONCE per batch (identical semantics to `ModernBertModel.forward`, sdpa/eager
  only — the flash path unpads internally and is not supported). Loss = LS-CE on the
  gate-blended probs (logsumexp-stable, fp32) + Switch load-balance aux (diffed vs HF
  transformers 4.51.3 `mixtral.load_balancing_loss_func`: identical with top_k=1, one
  routing decision per SEQUENCE) + usage-entropy floor on the batch-mean gate
  (`relu(tau*ln E − H(mean w))` — blocks global collapse, leaves per-row decisiveness
  free). Custom save/load (`save_moe`/`load_moe`, offline-safe via config skeleton).
- **`src/moe_finetune.py`** — champion-recipe entry point (richargs, LS ε=0.1, bf16,
  `--full_data`, same split calls as finetune.py INCLUDING string-label
  `split_indices(y)` — the E30 int-vs-string fold trap); imports `build_dataset` /
  `make_best_snapshot` / `make_compute_metrics` from finetune.py. Gate scalars from raw
  records via `GateFeatsDataset`/`GateFeatsCollator`. Post-train read-out pass caches
  `moe_val_parts.npz` (gate weights + per-expert log-probs) and prints blend / uniform
  null / solos / first-step slice / gate usage+entropy. `src/finetune.py` untouched.
- **Smoke evidence (local box, 2026-07-13):** tiny-config unit suite ALL PASS
  (grad reach to trunk/gate/all experts · blend matches reference & sums to 1 · aux ≈ 1
  and usage-H ≈ ln4 at init · save/load roundtrip exact · grad-checkpoint parity exact) ·
  **real-granite parity EXACT (max |Δlogp| = 0.0)**: with uniform gate + dense classifier
  copied into experts, every expert reproduces the dense 22L model bit-for-bit — masks/
  positions/final_norm placement verified · 555M params total (dense 312M) · GPU bf16
  forward peak 2.83 GiB (bs3@96) · end-to-end `moe_finetune.py --limit 24` CPU run
  exercises data→Trainer→eval→save→CSV.
- Training memory: measured-informed estimate ~15GB at champion bs4@512 (params+AdamW
  ≈ 8.9GB fp32 + ~3.2× the dense activations) → fits one 3090 WITHOUT grad
  checkpointing (`--grad_checkpointing on` = fallback, parity-verified exact).
  Wall-clock ≈ 3.2 pass-equivalents × champion full-FT (4–5h/3090) ≈ **13–16h on one
  3090**. v1 runs single-GPU; expert-parallel (one branch per GPU) would need custom
  device placement — only worth building if the queue needs the speed.
- Trunk receives gradient from all 4 experts + the gate (~4× the gradient traffic of any
  single expert's layers) — standard shared-bottom training, but watch the gate-entropy
  curve early to confirm the load-balance aux is holding.
- Log exact command + log path with the result; figure per result (gate-weight
  histogram + blend-vs-baselines bar at minimum).

## Stage-1 dispatch — 🏃 4 RUNS on vast, one per GPU (launched 2026-07-13)

**Instance:** vast `sandbox_4x_40gb` (ID 44587984, ssh1.vast.ai:27984) — **4× RTX 3090
24GB** (label "40gb" = the 40 vCPUs, NOT GPU mem; same stack as ocean). Code rsynced up
(the 2 files are uncommitted). Each run pinned to **one GPU** — single-GPU by design
(DataParallel would split the batch and corrupt the batch-level load-balance/entropy-floor
statistics). Micro-batch = 4 on every run so the per-forward gate stats stay comparable.

One run per GPU = the champion-recipe baseline PLUS the **E28 optuna top-3** tuned configs
(user 2026-07-13: use the idle GPUs with the better HPs). E28 found a higher-lr / eff_batch-8
/ heavier-smoothing corner (+0.0138 fold-0 F1 over champion); **epochs=5 there is a
best-epoch artifact, real peak is epoch 3** → all E28 runs use `--epochs 4` + `--full_data`
best-epoch snapshot (NOT `--all_data`, which shipped the E28 overfit tail → LB 0.75934).

| GPU | tag | recipe | lr | eff_batch | warmup | ε (LS) | wd | epochs |
|----|-----|--------|-----|:--:|:--:|:--:|:--:|:--:|
| 0 | `e33_moe_k6` | champion defaults (baseline — cleanest "does joint MoE help" test) | 2.0e-5 | 16 | 0.05 | 0.10 | 0.01 | 3 |
| 1 | `e33_moe_t043` | E28 #1 (fold-0 0.7724) | 2.60e-5 | 8 | 0.103 | 0.187 | 0.057 | 4 |
| 2 | `e33_moe_t048` | E28 #2 (0.7715) | 3.52e-5 | 8 | 0.037 | 0.174 | 0.059 | 4 |
| 3 | `e33_moe_t023` | E28 #3 (0.7709) | 4.35e-5 | 8 | 0.077 | 0.116 | 0.010 | 4 |

Common: granite, richargs, `--full_data`, bf16, trunk_k 6, 4 experts, balance 0.01,
entropy 0.01/τ0.75, `--batch_size 4` (eff_batch via `--grad_accum` 4 or 2). Logs
`/workspace/moe_e33{,_tNNN}.log`, sentinels `/workspace/DONE_e33{,_tNNN}`, read-outs →
each `<run_dir>/moe_val_parts.npz`.

- **All 4 confirmed live:** full_data 66.5k/3.5k · 555M · bf16 · **~13.6–13.9 GB/24** each ·
  all 4 GPUs training. GPU0 past warmup healthy (loss 10.4→5.5 by epoch 0.19, grad_norm
  100→35); t043/t023 start their transient lower (~5.3, gentler/heavier-smoothing schedules).
- **Why keep GPU0 (champion-default) too:** it's the apples-to-apples test of whether joint
  MoE training beats the 0.7803 champion / 0.78719 trio at the *same* recipe; the E28 runs
  add HP tuning (confounds "did MoE help") but give better members + ensemble diversity.
- **On finish:** rsync 4 run dirs back to NFS, compare each blend vs its own uniform-null
  vs solos + gate diagnostics, pick survivors, then package/idle per user.

## Results

All on the **3,500-row full_data holdout** (same split champion scored 0.7803 on → directly
comparable; slice mis-ranks, LB judges any final claim). "blend" = gate-weighted (the MoE);
"uniform" = mean of its own 4 experts, gate bypassed (the scientific null).

| run (recipe) | blend | uniform-null | best solo | Δ blend−champ 0.7803 | Δ blend−uniform | dead expert (argmax≈0) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `k6` (champion default) | **0.7702** | 0.7532 | 0.742 | **−0.0101** | +0.0170 | none (min 2.9%) |
| `t023` (E28 #3) | 0.7695 | 0.7624 | 0.731 | −0.0108 | +0.0071 | expert 2 (0.0%) |
| `t043` (E28 #1) | 0.7687 | 0.7600 | 0.754 | −0.0116 | +0.0087 | expert 3 (0.0%) |
| `t048` (E28 #2) | 0.7686 | 0.7312 | 0.731 | −0.0117 | +0.0374 | experts 2,3 (weak) |

Expert solos: k6 [.512,.615,.703,.742] · t023 [.559,.731,**.429**,.707] · t043 [.604,.716,.754,**.264**]
· t048 [.664,.731,.483,.332]. (All on the 3,500 full_data holdout; champion 0.7803 same holdout.)

### Verdict — E33 CLOSED, clean NEGATIVE (2026-07-13)

- **Joint MoE does NOT beat the champion single model.** All 4 configs land **0.7686–0.7702,
  uniformly −0.010 to −0.012 below champion 0.7803** (and far below trio 0.78719). The
  shared-trunk architecture is the ceiling: each expert = granite's top-16 layers on a
  shared 6-layer trunk with gradient split 4 ways → individually weaker (best solo ~0.75)
  than a standalone granite; the blend recovers most but not all of the gap. Matches the
  stated prior (MoE = cap-fit play, not accuracy-ceiling play).
- **E28 hyperparameter tuning did NOT transfer.** The tuned corner that gave +0.0138 on the
  *dense* granite gave the MoE runs blends *equal-to-slightly-worse* than champion defaults
  (t043/t048/t023 ≤ k6). The bottleneck is architecture, not recipe.
- **The gate consistently beats its own uniform-null (+0.007…+0.037) — but by EXPERT
  PRUNING, not specialization.** This *does* differ from E29 (frozen-member gates always
  lost to uniform): a *jointly-trained* gate can beat the mean — because training splits the
  4 experts into useful + useless and the gate learns to route around the dead ones (in t023
  and t043 one expert gets argmax-share **exactly 0.0**). So the E33 mechanism = "co-train
  then prune," not "co-adapt into complementary experts." The uniform mean is only worse
  because it naively includes the dead experts the gate discards.
- **Partial expert death, not full collapse.** 4 experts is too many for this task — every
  run kills or cripples 1–2 experts (entropy 1.10–1.22 / 1.386). The load-balance aux +
  entropy floor prevented E29-④ single-expert collapse but not per-expert death. Effective
  ensemble ≈ 2–3 experts.
- **Ensembling read:** within-one-MoE averaging is capped (shared trunk → correlated + dead
  experts). Any residual value is as a *diverse architecture member* for the trio (decorrelated
  errors), but at ~7:15 inference it breaks the 10-min cap alongside the trio → not a free add.
  LB could differ (slice mis-ranks) but −0.011 below champion on the same holdout is not
  promising; no submission built (no marginal success vs champion). Models on NFS
  `output/moe_..._e33_moe_{k6,t043,t048,t023}/`.

Stage-2 knobs (trunk_k sweep, fewer experts, expert-features gate) NOT pursued — the clean
−0.011 floor across 4 configs + partial expert death make a within-architecture rescue
unlikely to clear champion; parked unless revived.

### Follow-up — expert disassembly + reuse screen (user ask, 2026-07-13)

Each MoE's experts are extractable as standalone 22-layer granite classifiers (shared emb +
trunk 0–5 + expert layers 6–21 + expert head/classifier). `experiments/moe/extract_experts.py`
does the weight surgery, parity-checks each against the MoE's own per-expert path
(max|Δlogp| ~1e-5, bit-exact), dumps **raw 3,500-holdout logits** per expert
(`output/moe_experts/<tag>_e<e>/val_logits_3500.npz` + combined `val_logits_3500_<tag>.npz`,
each with `va_idx`+`labels`), and saves the fp16 checkpoint. Default extracts only ALIVE
experts (cached solo mF1 ≥ 0.65) → **8 of 16** (dropped 8 dead/weak, solo 0.26–0.62).

**Ensemble screen** (`screen_experts.py`, honest — all 16 held out the same 3,500 rows,
uniform means only, no fitting):

| combo | mF1 | vs champion 0.7803 |
|---|:--:|:--:|
| best single expert (`t043_e2`) | 0.7540 | −0.0263 |
| mean top-expert of each MoE (n=4) | 0.7632 | −0.0171 |
| **mean of 8 alive experts** | **0.7744** | **−0.0059** |
| mean all 16 | 0.7711 | −0.0092 |

Error diversity among strong experts: mean pairwise error-corr **0.605** (trio-comparable
~0.5–0.6); notably WITHIN-MoE pairs are MORE decorrelated (0.523) than cross-MoE (0.618) —
the gate did push experts to specialize. **But the best honest expert-ensemble (0.7744) is
still −0.006 below the champion single model and −0.013 below the trio**, and running 8
granites blows the 10-min cap → not a standalone submission. Residual value = diverse members
for the trio (needs an OOF harvest to test honestly, since the trio saw these 3,500 rows;
faces the same inference-cap wall). Logits banked for that future test. Artifacts:
`output/moe_experts/` (8 ckpts + per-expert & combined 3.5k-holdout logits + `manifest.jsonl`).
