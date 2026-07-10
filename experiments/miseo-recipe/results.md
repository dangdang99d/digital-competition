# Group: miseo-recipe — reproduce the teammate's training recipe on our pipeline
Logical group `research/miseo-recipe` (no git branch — shared working tree). Baseline anchors: **his LB 0.77427** (names, 2-fold ensemble + OOF bias) · **his OOF base 0.7642** (pre-bias) · **E21 names+LS+full_data 0.7731** · **E8a richargs+LS+full_data 0.7803**. All our scores raw uncalibrated macro-F1 on the full_data 3.5k slice.

Source of truth for his recipe: `teammate_work/miseo_koen_v2/digital-competition-miseo_ko-en/build_nb.py` (obtained 2026-07-09). His settings that differ from our champion recipe: **plain CE** (no LS) · **warmup_ratio 0.1** (ours 0.05) · **fp16** (ours bf16) · **names serialization** (ours richargs). Identical already: granite-311m-multilingual-r2, lr 2e-5, 3 epochs, effective batch 16, weight decay 0.01, max_len 512, hist cap 12, best-epoch by val macro-F1, seed 42.

Split note: per user instruction both arms use OUR `--full_data` split (66.5k train / 3.5k val slice) as the stand-in for his train-on-all mode (`FOLD="all"` in his notebook) — NOT his StratifiedGroupKFold (that protocolV is implemented as `--session_fold`, available but unused here). Consequence: E25a is a recipe repro, not a split repro — his OOF numbers (leak-free, 80% data) are not directly comparable to our slice numbers; same-slice deltas (E25a↔E25b↔E21↔E8a) are the trustworthy read-outs.

## Summary

| Exp | Arm | Serialize | Loss | Warmup | Precision | Status | Result | Verdict |
|---|---|---|---|---|---|:--:|---|---|
| E25a | his recipe verbatim | names | CE | **0.1** | fp16 | ✅ | **0.7692** | repro sane (expected ~0.76x) |
| E25b | his recipe + our levers | names | LS ε=0.1 | **0.1** | bf16 | ✅ | **0.7742** | LS+bf16 = **+0.0050** on his recipe |
| E25c | = b, serialization → richmeta | **richmeta** | LS ε=0.1 | **0.1** | bf16 | ✅ | **0.7801** | our packaging > his: **+0.0059** vs E25b; **≈ E8a richargs 0.7803** |
| *E21* (anchor) | prior run, = E25b except warmup | names | LS ε=0.1 | *0.05* | bf16 | ✅ | 0.7731 | vs E25b: warmup 0.1 = **+0.0011** (≈noise) |
| *E8a+LS* (anchor) | champion | richargs | LS ε=0.1 | *0.05* | bf16 | ✅ | **0.7803** | vs E25c: richmeta+warmup0.1 ties (−0.0002, 2-axis) |

All E25 arms ran warmup_ratio **0.1** (his value); both anchors were trained at our default **0.05** — every cross-table comparison involving an anchor carries the warmup axis (~+0.001) alongside the named one.

## Design

- **E25a — repro:** every axis at his value. If our pipeline faithfully implements his recipe, this should land near his level (expected ≈ E21 0.7731 minus the LS effect ≈ 0.76x on our slice; E9 measured LS worth +0.0107 on granite).
- **E25b — stack:** his recipe + LS + bf16 (the two champion levers he lacks).
- **E25c — path style (added mid-run, user 07-09):** = E25b but `--serialize richmeta`. The requested "our serialization with only his history/meta path style" is **exactly the existing `richmeta` variant** (our structure + full history arg paths; basenamed open-files meta already shared with his format) — no new variant created, existing variants verified byte-identical to git HEAD (0/70000 × richmeta/richargs/v1). Direct test of his relayed "richmeta > richargs" claim under this recipe.
- **Read-outs:**
  1. `E25b − E25a` = what **LS + bf16** add on top of his recipe (bundled by design; E9 says LS dominates).
  2. `E25b vs E21 (0.7731)` = **warmup 0.1 vs 0.05** as a clean single axis (only difference between the two runs).
  3. `E25a vs anchors` = sanity that his recipe reproduces in our code path.
  4. `E25c vs E25b` = our packaging vs his packaging of the SAME information (paths identical, format differs). `E25c vs E8a 0.7803` = richmeta+warmup0.1 vs richargs+warmup0.05 (two axes; if E25c wins, a richargs+warmup0.1 twin isolates serialization).
- **Residual differences from his true runs (documented, accepted):** split (ours full_data slice vs his train-on-all/OOF), transformers 4.51.3 vs his 4.48.3, single-GPU vs his DataParallel, no OOF logit-bias and no fold ensemble (NO-calibration project decision), batch 4×accum 4 vs his 16×1 (same effective batch, same math).

## Commands

```
bash sbatch/e25_miseo_recipe.sh a <gpu>   # E25a repro  -> sbatch/logs/e25a_miseo_repro.out
bash sbatch/e25_miseo_recipe.sh b <gpu>   # E25b stack  -> sbatch/logs/e25b_miseo_ls_bf16.out
```

New plumbing (2026-07-09): `--warmup_ratio` · `--precision auto|bf16|fp16` · `--session_fold`/`--session_splits` (leak-free SGKF, unused in E25) · serialization variants `names_files` + `richfiles` (his HISTPATH="files" surgical arg-stripping; available for a follow-up serialization arm).

## E25a — his recipe verbatim
- **Status:** ✅ done 2026-07-09 (~1h50m, GPU 0). Log `sbatch/logs/e25a_miseo_repro.out`.
- **Result:** **0.7692** (best = epoch 2; epoch 3 declined to 0.7655, best-checkpoint selection took epoch 2). Per-epoch: 0.7123 → **0.7692** → 0.7655.
- **Read:** his recipe reproduces sanely on our pipeline — lands right where the anchors predict (E21 0.7731 − LS effect). His leak-free OOF base 0.7642 is below this, consistent with our slice being easier. No evidence of a hidden recipe edge.

## E25b — his recipe + LS + bf16
- **Status:** ✅ done 2026-07-09 (GPU 1). Log `sbatch/logs/e25b_miseo_ls_bf16.out`.
- **Result:** **0.7742** (per-epoch 0.7156 → 0.7702 → **0.7742**, monotone — LS keeps epoch 3 improving where CE regressed).
- **Read:** ① LS+bf16 add **+0.0050** on his recipe (E25b−E25a). ② vs E21 0.7731 (identical but warmup 0.05): warmup 0.1 worth **+0.0011** — marginal, borderline noise.

## E25c — E25b with richmeta (our format, his full-path history style)
- **Status:** ✅ done 2026-07-09 (GPU 2). Log `sbatch/logs/e25c_richmeta_ls_bf16.out`.
- **Result:** **0.7801** (per-epoch 0.7221 → 0.7704 → **0.7801**).
- **Read:** ① vs E25b (+0.0059): with path information identical, **our multi-line packaging beats his one-line format** — confirms E21's direction under his own recipe settings. ② vs E8a richargs+warmup0.05 **0.7803**: dead tie (−0.0002); given warmup ≈ +0.001 (read-out ②/E25b), richmeta alone ≈ richargs or a hair below. **His "richmeta > richargs" claim does NOT reproduce on our slice** — they are statistically equal here, and the only LB pair we have (bge-m3) favored richargs (+0.0022). A `richargs + warmup 0.1` twin would fully isolate the axis but is unlikely to change the conclusion.

## Verdict
✅ **Done — no recipe edge found; champion recipe stands.** His training recipe reproduces but does not beat ours anywhere: our levers (LS+bf16) improve HIS recipe (+0.005), our serializations beat his format under his own settings (+0.006), and richmeta ≈ richargs (0.7801 vs 0.7803) — his relayed richmeta>richargs claim is not confirmed. Warmup 0.1 is worth ≤ +0.001 (optional, adopt-if-free). E8a granite richargs+LS 0.7803 / TTA LB SOTA unchanged.
