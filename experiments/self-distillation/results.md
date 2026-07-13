# Branch: self-distillation — same-model knowledge transfer for accuracy
Branch: `kyusang_kvprune_svd` · logical group `research/self-distillation` (per-branch topology planned but never created — all work committed on kyusang_kvprune_svd). All scores uncalibrated macro-F1.

**Baseline is the LS recipe, not plain CE.** Self-distillation is, by the Tf-KD result ([Yuan et al., CVPR'20](https://arxiv.org/abs/1909.11723)), *adaptive label smoothing* — standard KD ≡ a learned, per-sample LS; plain LS is the degenerate case with a fixed uniform teacher. E9 already promoted LS ε=0.1 (+0.0107, in the LB-SOTA recipe). So the bar here is **beat LS, not beat CE** — anything that only matches LS is redundant. Concretely: granite LS = **0.7565** (E9), qwen3+LS LB = **0.77921** (E8b+LS, champion).

## Why this lane exists (and what's NOT already covered)
The queue has two distillation items, **both cross-model, neither same-size self-distillation for accuracy**:
- **E10** ensemble→single (multi-teacher → one student) — 🕐 deferred to pre-deadline.
- **E4b** Minitron distill-recovery (unpruned parent → *pruned* child) — compression-recovery, gated on E4.

The gap: **same-architecture self-distillation for accuracy** — the model teaches itself from its own past predictions / a prior generation / its own deeper layers. E9's LS sits at the uniform-teacher end of this exact spectrum; this lane tests the *informative-dark-knowledge* end.

## Priors that shape expectations (be honest up front)
- **Label-softening family has WON here; consistency family has LOST.** LS won (E9). supcon −0.7pt, R-Drop stalled — both consistency regularizers. ⇒ prefer target-softening self-KD (PS-KD, BAN) over consistency self-KD (DLB, CS-KD).
- **Residual error is intrinsic label ambiguity** (ceiling/specialist/capacity probes all saturate; reasoning caps ~0.28). Self-distillation is a *regularizer* — it cannot reduce Bayes error. Expect an LS-magnitude gain **at best**, and a real chance of null (like logit-adjustment τ=1 was −0.0002). This is a cheap flyer, not a headline lever.
- **CV→LB reversal (E8):** the granite 3.5k-slice CV mis-ranked qwen3. Treat any granite screen as a *filter* (does it beat LS at all?), never a ranking; confirm winners on qwen3 + LB.
- **Invariant — get the official code first.** Every arm below lists the authors' repo; diff our implementation against it before trusting numbers (papers omit temperature/α schedules, detach placement, sampler details).

## Summary
Legend: ✅ win · ❌ closed · 🟡 low-prio · 🕐 deferred · 📋 planned (not run).

| Exp | Method | Venue | Teacher source | Model | Extra inf. cost | Prior | Status | Result | Δ vs LS |
|---|---|---|---|:--:|:--:|:--:|:--:|---|---|
| E19a | **PS-KD** | ICCV'21 | own, last epoch | granite→qwen3 | 0 | ✅ soften | 📋 | — | — |
| E19b | **BAN** | ICML'18 | prior generation | granite→qwen3 | 0 (or ens.) | ✅ reg | 📋 | — | — |
| E19c | **BYOT** | ICCV'19 | deeper layer | qwen3 (depth-redundant) | 0 / faster | ✅ deep-sup | 📋 | — | — |
| E19d | **DLB / CS-KD** | CVPR'22 / '20 | last iter / same-class | granite | 0 | ➖ consistency | 📋 | — | — |
| — | Tf-KD_reg | CVPR'20 | hand-designed const | — | 0 | = LS | ⛔ | ≡ E9 LS | — |

Through-line: all replace the one-hot's zero off-target mass with a softer distribution, differing only in its **source** — uniform (LS, done) → own past epoch (PS-KD) → own last iter (DLB) → same-class sibling (CS-KD) → prior generation (BAN) → deeper layer (BYOT) → hand-designed constant (Tf-KD_reg ≡ LS).

---

## E19a — PS-KD (Progressive Self-Knowledge Distillation) — PRIMARY
- **Method:** [Kim et al., ICCV'21](https://arxiv.org/abs/2006.12000) · official code: [lgcnsai/PS-KD-Pytorch](https://github.com/lgcnsai/PS-KD-Pytorch).
- **Model:** granite-311m screen → qwen3-0.6B confirm.
- **What:** the model's own softmax from the **previous epoch** is the teacher; weight it lightly early, ramp up ("progressive"). This is LS with the uniform `u` replaced by informative self-predictions.
- **Mechanism:** refined target `ŷ_t = (1−α_t)·y_onehot + α_t·p_{t−1}(x)`, loss `CE(ŷ_t, p_t)`; `α_t` grows over epochs (linear 0→~0.8). Cache last-epoch probs (`N×K`, tiny), ~0 extra compute.
- **Baseline:** the **LS** recipe on the same backbone (granite LS 0.7565; qwen3+LS 0.77921 LB). Not CE — CE would be an unfair, already-beaten bar.
- **Read-out:** granite arm > LS +0.003 → promote to qwen3 + build LB zip; ≤ LS → CLOSE (informative dark-knowledge ≯ uniform for weak-semantic labels).
- **Fit:** cheapest arm, drops into the existing `--loss` hook, stacks on LS, in the family that won. **Do this first.**
- **Status:** 📋 planned. **Result:** —

## E19b — BAN (Born-Again Networks) — the strong shot
- **Method:** [Furlanello et al., ICML'18](https://arxiv.org/abs/1805.04770) · no single official repo — pick the most-starred faithful reproduction and record which in the Result.
- **Model:** granite screen → qwen3.
- **What:** train `M_0` from labels → freeze as teacher → train `M_1` (*fresh init, identical arch*) on `M_0`'s soft labels → repeat `k` generations. Take last gen, or **ensemble the generations** (usually strongest — feeds E12 soup).
- **Mechanism:** gen `i`: `CE(y, p_i) + KD(p_{i−1} → p_i)`, temperature T (sweep {2,4}). Diagnostic ablations from the paper (CWTM = teacher-max reweight only; DKPP = permute non-argmax) attribute the gain largely to regularization/reweighting, not similarity transfer — matches our weak-semantic-labels theme.
- **Baseline:** LS recipe (the `M_0` should itself be an LS model so gen-1 is a fair test); also report vs E12 soup if generations are ensembled.
- **Read-out:** BAN-gen1 or gen-ensemble > LS +0.003 → promote; the ensemble variant must beat E12 soup to justify itself.
- **Cost:** k× training (~2-3× GPU). **Overlap with E12:** BAN generations are natural soup ingredients — coordinate.
- **Status:** 📋 planned. **Result:** —

## E19c — BYOT (Be Your Own Teacher) — accuracy + speed knob
- **Method:** [Zhang et al., ICCV'19](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.html).
- **Model:** qwen3-0.6B (its deep stack is redundant — E16 depth-prune was ~free; BYOT exploits the same redundancy). granite is a poor fit (E16b: each layer load-bearing).
- **What:** attach small pooled classifiers after intermediate blocks; the **deepest exit** distills *down into* the shallow ones during one training run (no pre-trained teacher).
- **Mechanism:** per shallow exit — (1) CE to true label, (2) KL from deepest exit (softened), (3) L2 hint on features. Deploy the deepest exit (full acc, 0 overhead) OR adaptive **early-exit** for speed — the shallow exits are *learned* early-exit points.
- **Baseline:** qwen3+LS (accuracy) and the E16/E17/E18 speed anchors (inference time).
- **Read-out:** deepest exit ≥ qwen3+LS (accuracy neutral-or-better) AND early-exit gives a real ms/sample win vs the E16 depth-prune anchor → adopt for the speed track.
- **Engineering caveat:** custom forward + exit heads must survive `from_pretrained` in the ≤1GB zip — same class of work as E4 (factored FFN) / E18 (token prune). Plan the load-hook up front.
- **Fit:** the only arm that couples accuracy with the 본선 speed score (10%) — but heaviest. Do only if the goal is speed, not pure accuracy.
- **Status:** 📋 planned. **Result:** —

## E19d — DLB / CS-KD (consistency self-KD) — low prior
- **Method:** DLB [Shen et al., CVPR'22](https://arxiv.org/abs/2203.16172), official [Meta-knowledge-Lab/DLB](https://github.com/Meta-knowledge-Lab/DLB) · CS-KD [Yun et al., CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/html/Yun_Regularizing_Class-Wise_Predictions_via_Self-Knowledge_Distillation_CVPR_2020_paper.html).
- **Model:** granite (screen only).
- **What:** DLB — overlap half of each mini-batch with the previous iteration, distill on prior-iter soft targets. CS-KD — force same-class samples to agree (`KL(stopgrad p(x') ‖ p(x))`).
- **Baseline:** granite+LS.
- **Read-out:** > LS +0.003 → reconsider; else CLOSE.
- **Prior:** ➖ **consistency family, which has underperformed here** (supcon −0.7pt, R-Drop stalled). Lower expectation than PS-KD/BAN. Run only if PS-KD is promising and a cheap consistency comparison is wanted.
- **Status:** 📋 planned (deprioritized). **Result:** —

## Tf-KD_reg — NOT a new arm (bookkeeping)
[Yuan et al., CVPR'20](https://arxiv.org/abs/1909.11723) has two variants: `Tf-KD_self` (own pretrained model as teacher ≈ 1-generation BAN → covered by E19b) and `Tf-KD_reg` (hand-designed high-prob-correct + uniform-elsewhere target at temperature T ≈ **label smoothing** → covered by E9). ⛔ Not run separately; listed so it isn't re-proposed.

## Results log
(append: date · arm · key numbers · memory file)
- (none yet — lane is planned; E19a PS-KD is the proposed first run.)
