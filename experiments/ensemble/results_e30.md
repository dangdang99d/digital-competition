# Branch: ensemble-oof (E30) — session-grouped OOF campaign + honest-teacher distillation retry

**STATUS: ✅ DONE (2026-07-12).** E26 continuation ([results_e26.md](results_e26.md): trio LB
0.78719 = direct-package ceiling). Phase 0 ✅ — honest OOF caches for all 70k rows (first
harvest pass was invalidated by a fold-construction mismatch and fully re-run; validity check:
member solo OOF mF1 0.764–0.767 = the fold-models' own logs). Phase 1 ❌ — honest-teacher
students all below champion (best 0.7753, −0.0050): **distillation closed; the trio's edge is
inference-time averaging, not distillable.** The caches also powered E29 (learned heads — all
arms failed to beat uniform mean; see [results_e29.md](results_e29.md)).

## Method

Goal: for every training row, get each trio member's prediction **from a model that never saw
that row** (out-of-fold). The deployed members can't provide this — they trained on ~all 70k,
so their outputs are memorized near-one-hots (what sank E26-1B).

1. **Fold:** split the 70k into 5 folds grouped by SESSION (StratifiedGroupKFold, seed 42,
   zero overlap asserted) — row-level folds leak via session siblings.
2. **Train:** per member recipe (is3 / aum06 / e25c, exact original hyperparameters,
   eff. batch 16), train 5 fold-models from scratch on 4/5 of the data → 15 trainings.
3. **Harvest:** run each fold-model on its held-out ~14k rows; save the 14-class softmax
   (fp32) + penultimate embedding (fp16, input of the final classifier layer).
4. **Merge:** stitch folds → per member, honest probs + embedding for all 70k rows
   (`e30_oof_<member>.npz`; coverage asserted).
5. **Teacher:** per-row uniform mean of the 3 members' OOF probs — the OOF analogue of the
   LB trio. Ambiguity → spread mass (dark knowledge); wrong label → teacher contradicts it.
   Abort guard: mean MSP ≥ 0.97 = in-sample signature.
6. **Students (phase 1):** from-scratch champion recipe, loss
   `(1−α)·CE_LS + α·T²·KL(teacher_T ‖ student_T)` (α=0.7, T=2) — CE keeps label smoothing,
   the ≈0.0107 term 1B silently dropped.

Scripts: `oof_harvest.py` (1–4) · `build_oof_teacher.py` (5) · `finetune.py --distill_from
--loss ls` (6).

## Phase 0 results — ✅ COMPLETE (2026-07-12; ~3.5h wall on 4×3090 incl. one full re-harvest, ≈$2)

15/15 fold-trainings (losses nominal, no NaN) + harvests + full-coverage merges (70,000 rows ×
probs + 768-d emb per member). ⚠️ Methodological caveat that forced a re-harvest: fold
construction must byte-match the training splits — the first pass diverged (~20% fold overlap
→ ~80% in-sample), detected because member "OOF" mF1 read 0.83 vs the fold-models' own
0.76–0.77 logs. Honest-pass validity checks: solo OOF mF1 is3 0.7668 · aum06 0.7644 ·
e25c 0.7664 ✓.

| teacher | MSP mean | MSP p95 | note |
|---|---|---|---|
| trio (is3+aum06+e25c) | **0.738** | 0.912 | genuinely soft — 1B's in-sample teachers were ~0.99 |
| is3 single (born-again control) | 0.733 | 0.913 | 〃 |

Honest by-product: uniform mean over the three OOF prob sets scores **0.7745** vs solos
0.764–0.767 — the +0.008 ensemble lift reproduces the LB gain (+0.0079) on 70k honest rows.

## Phase 1 (OOF-teacher students) — ❌ CLOSED (honest run 2026-07-12)

Four students (mirror the E26-1B decomposition; one wave on 4 GPUs), all from-scratch
champion recipe + full_data, eval on the shared 3.5k slice:

| # | teacher | α | T | LS | tests |
|---|---|---|---|---|---|
| s1 | OOF trio | 0.7 | 2 | ✓ | workhorse (1B-#1 analogue) |
| s2 | OOF trio | 1.0 | 1 | (moot at α=1) | teacher-as-labels; 1B's worst arm — sign flip = harvest was the problem |
| s3 | OOF trio | 0.5 | 2 | ✓ | mix-ratio dose-response |
| s4 | OOF is3 only | 0.7 | 2 | ✓ | born-again control (ensemble knowledge vs KD-regularizer) |

(1B's 5-member qwen3 teacher has no analogue — no OOF caches for champion/qwen3; +10 trainings
if ever wanted.) Gate: any student ≥ 0.7803 → member pool → re-run E26 greedy selection.

**Results (honest teacher, MSP 0.749; all runs clean):**

| # | teacher / setting | slice mF1 (Δ vs champion 0.7803) | 1B in-sample twin |
|---|---|---|---|
| s3 | trio OOF · α .5 · T2 · +LS | **0.7753 (−0.0050)** | — |
| s4 | is3 OOF (born-again) · α .7 · +LS | 0.7722 (−0.0081) | 0.7728 |
| s1 | trio OOF · α .7 · T2 · +LS | 0.7697 (−0.0106) | 0.7691 |
| s2 | trio OOF · α 1 · T1 (pure soft) | 0.7680 (−0.0123) | 0.7635 |

**Verdict — phase 1 ❌ CLOSED (valid this time).** With a genuinely honest teacher the
students still all land below the champion; teacher weight is monotone-harmful at the top of
the curve (α .5 > .7 ≫ 1.0, and α=0 — the plain LS champion — beats them all). The teacher's
spread mass sits on intrinsically ambiguous rows (E19 prior) and carries no separating signal
beyond labels+LS: **the trio's edge is inference-time averaging of diverse errors, which
compression into one model removes by construction.** Ensemble→single distillation on this
task is closed — measured under in-sample teachers (1B), contaminated-OOF teachers (struck
first wave: 0.7754/0.7741/0.7726/0.7662), and honest-OOF teachers (above).

## Hand-off

- **E29** (learned head over the trio) consumes `e30_oof_*.npz` — gated on the user's
  no-calibration ruling. See EXPERIMENTS.md §E29.
- CL/PVI re-audit under champion recipes: free CPU pass over the merged caches (diagnostics
  only; denoise stays closed per E22).
