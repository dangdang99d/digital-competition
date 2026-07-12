# Branch: ensemble-oof (E30) — session-grouped OOF campaign + honest-teacher distillation retry

**STATUS: ✅ DONE (2026-07-12).** E26 continuation ([results_e26.md](results_e26.md): trio LB
0.78719 = direct-package ceiling). Phase 0 ✅ — 15/15 fold-trainings, full-coverage OOF caches,
**teacher MSP 0.749** (honest dark knowledge; 1B's in-sample teachers were ~0.99). Phase 1 ❌ —
**distillation closed for good**: every OOF student beat its 1B twin yet ALL landed below the
champion (best 0.7754, −0.0049); the trio's edge is inference-time averaging, not distillable.
Remaining value = the OOF caches → **E29** learned head + CL/PVI re-audit.

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

## Phase 0 results — ✅ COMPLETE (2026-07-12, ~3h wall on 4×3090, ≈$1.5)

15/15 fold-trainings + harvests, losses nominal (1.23 → ~0.94 by ep 1), no NaN; all
merges full-coverage (70,000 rows × probs + 768-d emb per member).

**Teacher softness — the OOF thesis confirmed:**

| teacher | MSP mean | MSP p95 | vs 1B in-sample teachers |
|---|---|---|---|
| trio (is3+aum06+e25c) | **0.749** | 0.912 | ~0.99 everywhere → real dark knowledge recovered |
| is3 single (born-again control) | 0.742 | 0.914 | 〃 |

## Phase 1 (OOF-teacher students) — ❌ CLOSED (run 2026-07-12)

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

**Results (2026-07-12, all clean):**

| # | teacher / setting | slice mF1 (Δ vs champion 0.7803) | vs 1B twin |
|---|---|---|---|
| s1 | trio OOF · α .7 · T2 · +LS | **0.7754 (−0.0049)** | +0.0063 |
| s3 | trio OOF · α .5 · T2 · +LS | 0.7741 (−0.0062) | — |
| s4 | is3 OOF (born-again) · α .7 · +LS | 0.7726 (−0.0077) | ±0.000 |
| s2 | trio OOF · α 1 · T1 (pure soft) | 0.7662 (−0.0141) | +0.0027 |

**Verdict — phase 1 ❌ CLOSED, this time with the real answer.** Honest teachers fixed what the
1B post-mortem predicted (every OOF arm ≥ its in-sample twin; s2's sign flip = harvest honesty
confirmed), yet every student stays below the champion, and the α dose-response is monotone
toward α=0: ANY teacher signal added to the champion recipe hurts. With teacher softness now
genuine (MSP 0.749), the conclusion is structural, not procedural: **the trio's edge is not
distillable into a single granite-311m** — the teacher's spread mass sits on intrinsically
ambiguous rows (E19 prior), carrying no separating signal beyond labels+LS; the ensemble's
value is inference-time averaging of diverse errors, which compression removes by construction.
Distillation on this task is closed for good (in-sample AND honest variants both measured).
E30's remaining value = the OOF caches themselves → E29 learned head + CL/PVI re-audit.

## Hand-off

- **E29** (learned head over the trio) consumes `e30_oof_*.npz` — gated on the user's
  no-calibration ruling. See EXPERIMENTS.md §E29.
- CL/PVI re-audit under champion recipes: free CPU pass over the merged caches (diagnostics
  only; denoise stays closed per E22).
