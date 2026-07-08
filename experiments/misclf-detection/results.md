# Branch: misclf-detection — detect when the frozen classifier is wrong

Branch: `research/misclf-detection`. Scope: **misclassification detection** (a.k.a. failure
prediction / error detection). For each input we predict whether the *already-trained, frozen*
champion's argmax is **wrong** — a scalar "wrongness"/confidence score per sample.

**Every detector attaches to the existing model without retraining it.** The classifier's
weights and argmax are untouched; the detector is a read-off (or a small parallel head) hung on
one of three tap points in the frozen forward pass. Downstream use (routing wrong-predicted
samples to a fallback) is a *later* concern — here we only build and rank detectors.

Baseline to beat: single-model margin→correctness **AUROC ≈ 0.83** (measured: qwen3 0.848).

**Base model (the classifier we detect errors on): original `qwen3`** — Qwen3-0.6B full-FT
v1@512 = 0.7682 uncal, the project's canonical champion baseline (logits already cached at
`analysis/cache/qwen3_val_logits.npz`, so M1/M5 run immediately). Re-extract the substrate on the
final model once the loss screen settles.

---

## The frozen model & its tap points

Every method below hooks one of these three points. Blue = the frozen classifier; nothing here
is retrained.

```mermaid
flowchart LR
    X["input<br/>(serialized sample)"] --> ENC["transformer<br/>× L layers"] --> H["h<br/>penultimate<br/>features ②"] --> HEAD["classifier<br/>head W"] --> Z["logits z<br/>(14) ①"] --> P["softmax p ①"] --> YHAT["argmax ŷ"]
    classDef frozen fill:#eef4fc,stroke:#9bbce8,color:#333;
    class X,ENC,H,HEAD,Z,P,YHAT frozen;
```

- **① output tap** — the logits `z` / softmax `p` (post-head). Cheapest; already cached.
- **② feature tap** — the penultimate pooled vector `h`.
- **③ stochastic tap** — re-run the encoder with dropout ON (M4 only).

---

## Per-method attachment

Blue = frozen classifier (unchanged) · solid arrows = the frozen forward pass ·
dotted arrows = where the detector taps in · orange = the detector (new).

### M1 · Output-based scores — tap ① (FREE)
A scalar function of the logits/softmax. No forward pass, no training — pure math on cached `z`.
```mermaid
flowchart LR
    ENC["frozen model"] --> Z["logits z ①"] --> P["softmax p"] --> YHAT["argmax ŷ"]
    Z -. tap .-> S["score s(x)"]
    P -. tap .-> S
    S --> OUT["wrongness score"]
    classDef frozen fill:#eef4fc,stroke:#9bbce8,color:#333;
    classDef det fill:#e8873c,color:#fff,stroke:#c9711f;
    class ENC,Z,P,YHAT frozen;
    class S,OUT det;
```
Scores computed on `z`/`p`: **MSP** `max p` · **entropy** `−Σ p log p` · **margin** `z₍₁₎−z₍₂₎` ·
**max-logit** `max z` · **energy** `−log Σ e^z` · **DOCTOR** `Σ p²`.

### M2 · Distance / density — tap ② (needs feature pass)
Score = how far `h` sits from the training class manifolds (stored offline from a train pass).
```mermaid
flowchart LR
    ENC["frozen model"] --> H["h · penultimate ②"] --> HEAD["head → ŷ (frozen)"]
    H -. tap .-> DIST["distance to<br/>class manifolds"]
    TRAIN["train-set stats<br/>(per-class means /<br/>covariance / kNN index)"] --> DIST
    DIST --> OUT["wrongness score"]
    classDef frozen fill:#eef4fc,stroke:#9bbce8,color:#333;
    classDef det fill:#e8873c,color:#fff,stroke:#c9711f;
    class ENC,H,HEAD frozen;
    class DIST,TRAIN,OUT det;
```
Variants: **Trust Score** (same- vs other-class NN distance ratio) · **Mahalanobis** (per-class
Gaussian) · **kNN label-agreement**.

### M3 · ConfidNet — tap ②, a parallel trained head
A small MLP `g_φ(h)` runs *alongside* the frozen head, trained (head frozen) to regress the
True-Class-Probability `TCP = p[y*]`. Only `g_φ` learns; the classifier never moves.
```mermaid
flowchart LR
    ENC["frozen model"] --> H["h · penultimate ②"] --> HEAD["head (frozen)"] --> Z["z → ŷ"]
    H -. tap .-> G["ConfidNet g_φ<br/>small MLP (trained)"]
    G --> C["ĉ ∈ [0,1]"]
    Z -. "train target:<br/>match TCP = p[y*]" .-> G
    classDef frozen fill:#eef4fc,stroke:#9bbce8,color:#333;
    classDef det fill:#e8873c,color:#fff,stroke:#c9711f;
    class ENC,H,HEAD,Z frozen;
    class G,C det;
```
Ref: valeoai/ConfidNet — get the official code, diff the TCP target/loss before trusting it.

### M4 · MC-dropout — tap ③ (N× inference, no training)
Turn dropout ON at inference, run the frozen model `N` times, score = disagreement across passes.
```mermaid
flowchart LR
    X["input"] --> M["frozen model<br/>(dropout ON) ③"]
    M --> R["N stochastic passes<br/>{p⁽¹⁾ … p⁽ᴺ⁾}"]
    R --> V["predictive variance /<br/>mean entropy"]
    V --> OUT["wrongness score"]
    classDef frozen fill:#eef4fc,stroke:#9bbce8,color:#333;
    classDef det fill:#e8873c,color:#fff,stroke:#c9711f;
    class X,M frozen;
    class R,V,OUT det;
```

### M5 · Cross-model disagreement — two frozen models (FREE)
The base qwen3 + a second independently-trained frozen model; when their argmaxes disagree, the
primary is far likelier wrong. Directly exploits the decorrelated errors behind the 0.810
two-model oracle. Second model = bge-m3 hist0 (cached now → runnable immediately) or granite later.
```mermaid
flowchart LR
    X["input"] --> A["model A · qwen3 (base)<br/>(frozen)"] --> ZA["ŷ_A, margin_A"]
    X --> B["model B · hist0 / granite<br/>(frozen)"] --> ZB["ŷ_B, margin_B"]
    ZA -. tap .-> CMP["agree?<br/>+ combined margin"]
    ZB -. tap .-> CMP
    CMP --> OUT["wrongness score"]
    classDef frozen fill:#eef4fc,stroke:#9bbce8,color:#333;
    classDef det fill:#e8873c,color:#fff,stroke:#c9711f;
    class X,A,B,ZA,ZB frozen;
    class CMP,OUT det;
```

---

## Summary
Legend: ✅ win · ❌ no-go · ⚠️ broken impl · 🟢 not run. Base model = original qwen3 (val acc
0.7654, 3285/14000 wrong). Ran 2026-07-08. AUROC = correct-vs-wrong on 14k val; AURC lower=better.

| Exp | Detector (best variant) | Tap | Cost | Status | AUROC ↑ / AURC ↓ | Verdict |
|---|---|:--:|:--:|:--:|---|---|
| M0 | Substrate extraction (qwen3 base) | — | 1 fwd pass | ✅ | parity Δ0.031 · recon 0.011 | done — `misclf_substrate_qwen3.npz` |
| M1 | **DOCTOR / MSP / neg-entropy** | ① | free | ✅ | **0.8538 / 0.069** | ✅ **WINNER** — free, robust au(0.92)/first(0.79) |
| M1 | margin (prior baseline) | ① | free | ✅ | 0.8474 / 0.070 | reproduced the ~0.83 incumbent |
| M2 | knn_agree · trustscore · **mahalanobis** | ② | cheap | ⚠️ | 0.817 · 0.782 · **0.319** | ❌ below M1; **mahalanobis anti-correlated = impl broken** |
| M3 | ConfidNet (trained TCP head) | ② | cheap | ❌ | 0.8316 / 0.075 | ❌ **underperforms free MSP** — not worth training |
| M4 | MC-dropout / TTA variance | ③ | N× infer | 🟢 | — | deferred (later) |
| M5 | cross-model vs hist0 (sum_norm_margin · agree) | ①×2 | free | ❌ | 0.847 · 0.619 | ❌ **no gain** over single-model (hist0 too weak) |

**Headline:** best detector = plain max-softmax family (**DOCTOR/MSP/entropy ≈ 0.854 AUROC**),
free and robust across slices — confirms the project's ~0.83–0.85 correctness-detection ceiling.
Trained (ConfidNet), cross-model (hist0), and distance (M2) detectors **do not beat it**.
Figures: [overall](figures/m_auroc_overall.png) · [robustness](figures/m_robustness_slices.png) ·
[risk-coverage](figures/m_risk_coverage.png). Full numbers: `m_results.json`.

**Open follow-ups:** (a) M2 Mahalanobis broken (AUROC<0.5 → raw-feature conditioning / sign) — fix
or drop. (b) M5 used hist0 (weak bge); retry with **granite** as the 2nd model. (c) M4 not run.

### Exact commands
```
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python experiments/misclf-detection/m0_substrate.py \
  --ckpt output/pat/qwen3_champion_pruned --tokenizer Qwen/Qwen3-Embedding-0.6B \
  --remap output/pat/qwen3_champion_pruned/remap.npy --out analysis/cache/misclf_substrate_qwen3.npz
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python experiments/misclf-detection/run_eval.py
```
(qwen3 champion weights extracted from `submit_0703_qwen3_pruned.zip`; HF tokenizer + remap.)

## M0 — Substrate extraction
- **Model:** original **qwen3** (champion 0.7682, plain CE), frozen. Logits + preds are ALREADY
  cached (`analysis/cache/qwen3_val_logits.npz`; `val_logits.npz` has labels + `va` generator) →
  **M1 and M5 (with cached hist0) are runnable right now, no pass needed.**
- **Only new work:** one inference pass to add the **penultimate embedding `h`** (needed by
  M2/M3), plus the sim/au + first-step masks (reuse `split_indices(42)` + `va`).
- **Output:** `analysis/cache/misclf_substrate_qwen3.npz` — read by M2/M3.
- **Re-run later:** once the loss screen picks the final recipe, re-extract on that model — the
  detector ranking may shift because the loss reshapes confidence.

## Read-outs (every detector reports all)
1. **AUROC(correct vs wrong)** — ranking quality (the headline).
2. **AURC + risk-coverage curve** — competition-relevant (Traub et al., 2024: AUROC alone
   misleads). Figure → `figures/`.
3. **Robustness slices** (MANDATORY — the E6 lesson): au · first-step · sim · later-step. A
   detector that only ranks well on the sim/later majority is a DROP.
4. *(secondary, forward-looking)* downstream macro-F1 if the lowest-score fraction is routed to
   a fallback — honest 2-fold on any threshold.

## Notes / prior evidence
- **E6** ([deferral/results.md](../deferral/results.md)) tried the *routing* half with hand-tuned
  per-class τ + a bge fallback: +0.0047 overall but net-negative on au / first-step → DROP. This
  branch's novelty = *principled/learned* detectors + an honest bake-off; robustness slices carried
  over so we don't repeat the blind spot.
- Ceiling: margin→correctness AUROC ≈ 0.83–0.85; two-model oracle = 0.810 F1.
- Literature (bucket B): MSP (Hendrycks & Gimpel, ICLR'17) · Trust Score (Jiang et al., NeurIPS'18)
  · ConfidNet (Corbière et al., NeurIPS'19) · DOCTOR (Granese et al., NeurIPS'21) · MC-dropout
  (Gal & Ghahramani, ICML'16) · eval flaws (Traub et al., 2024).
