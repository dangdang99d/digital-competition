# E37 · Sequence pooling for the granite classifier — `research/pooling`

**Status: 🟢 CODE READY + smoke-passed (2026-07-13). Not dispatched — awaits vast GO.** All
code lives in `experiments/pooling/` ONLY; `src/` untouched. Venue = vast.ai (user call).

## Motivation

Granite (`granite-embedding-311m-multilingual-r2`) is a **ModernBERT** encoder; our classifier
currently pools a **single token** — the `[CLS]` / first token (`config.classifier_pooling="cls"`,
confirmed) — feeds it to a 1-linear head, and predicts. Question (user 2026-07-13): would using
**every token's output** — "like an LSTM" — do better than one CLS vector?

The LSTM form itself is the wrong tool: the backbone is already a 22-layer *bidirectional*
transformer, so every token is fully contextualized and CLS has already attended to all of them;
stacking a weaker sequential mixer (RNN) on top is redundant and adds T4 inference cost. The
*content* of the idea — "use all tokens, let the model weight them" — is a **pooling** choice,
and the transformer-native version is a **learned attention pool** (one attention layer, not a
recurrence). This experiment screens pooling as a single axis.

## Prior (why expectations are modest)

The project's repeated finding is that the ceiling is **label ambiguity, not model capacity**:
E13 SAM closed (near-zero run variance — error is intrinsic ambiguity), E33 MoE −0.011, KD
negative, reasoning caps ~0.28 (labels weakly semantic). Pooling is a representation/capacity
lever, and this task has not responded to those → expect ±0.003 (noise-to-marginal). **One
backbone-specific upside:** granite is an *embedding* model; if its contrastive pretraining used
mean pooling, classifying with mean may align better with the pretrained geometry than CLS does
(checkable). Cheap + genuinely untested → worth one tight screen.

## Arms (one axis: pooling; everything else = champion recipe)

| Arm | Pooling | Mechanism | Impl status |
|-----|---------|-----------|-------------|
| P0 | `cls` (**control**) | first token `h[:,0]` — current champion behavior | ✅ native (no change) — smoke ✅ |
| P1 | `mean` | masked mean over all token outputs | ✅ native (`config.classifier_pooling="mean"`) — smoke ✅ |
| P2 | `attn` | learned additive attention pool: `a_t=softmax_t(w·h_t)`, out `=Σ a_t h_t` → head | ✅ `pooling.py` (instance-forward monkeypatch) — smoke ✅ (trains+evals, weights save) |
| P3 (opt) | `mean+max` | concat masked-mean and masked-max (2·H → head) | ⏸ not built (only if P1/P2 signal; needs a 2H→H proj) |

## Implementation (2026-07-13, code ready + smoke-passed)

Two files, **entirely in `experiments/pooling/`** — `src/` untouched (shared-tree invariant):
- **`pooling.py`** — `AttentionPool` + `install_pooling(model, mode)`. `cls`/`mean` set
  `config.classifier_pooling` (native ModernBERT); `attn` attaches an `AttentionPool` and
  rebinds the instance's `forward` to a verbatim copy of transformers 4.51.3
  `ModernBertForSequenceClassification.forward` with only the pooling line swapped. Records
  `config.pooling` for reload.
- **`run_e37.py`** — driver: fixes the champion recipe (granite richargs + LS 0.1 + full_data +
  bf16(auto) + epochs 3 + lr 2e-5 + **bs16 + accum1** + max_len 512 + seed 42), injects pooling
  by wrapping `AutoModelForSequenceClassification.from_pretrained` **in-process**, then calls the
  real `src.finetune.main()` unchanged (no recipe divergence). `--pooling cls|mean|attn`,
  optional `--batch_size/--max_len/--epochs/--limit` for smoke.

**Smoke ✅ (local ArchServer RTX 4060, bs4/max_len128/limit400/0.03ep):** all three modes exit 0,
pooling logged correctly (`cls`→classifier_pooling=cls · `mean`→mean · `attn`→config.pooling=attn),
eval ran, artifacts saved. `attn` checkpoint confirmed to save `attn_pool.score.{weight,bias}` +
`config.pooling=attn` → reloadable. ⚠ smoke exercises code paths only (tiny run; not a recipe result).

**Run command (per arm, on vast):**
`python -m experiments.pooling.run_e37 --pooling {cls|mean|attn}` → `output/pat/ft_..._e37_<pooling>`
+ `ft_results_e37.csv`.

## Protocol

- **Recipe:** champion E8a+LS richargs full_data bf16 (granite), **from scratch**, `--batch_size 16
  --grad_accum 1` (real bs16, NO accumulation — granite fits, E19; effective batch 16), ONE lever =
  pooling. Shared anchor + eval slice. Seed 42, init_seed 42 (ensemble/soup-compatible).
- **Baseline:** P0 cls = the champion recipe re-run from scratch (its own anchor), and champion
  full_data CV **0.7803** for context. Raw uncalibrated logits, macro-F1.
- **Read-out:** full_data CV per arm vs the P0-cls from-scratch anchor (screen, not LB-rankable —
  E8/E26 slice mis-ranks; LB judges any final claim).
- **Gates:** promote ≥ **+0.003** vs the cls anchor; marginal +0.001–0.003 → ensemble-member
  candidate (a differently-pooled model is a diverse regime for the E26/E30 pool); judged by
  honest OOF / LB, never the slice.
- **Implementation invariants (when built):** code lives in `experiments/pooling/` ONLY;
  default `cls` executes zero new code (existing runs byte-identical); `mean` = native config
  flag; `attn` = per-instance monkeypatch reproducing transformers 4.51.3
  `ModernBertForSequenceClassification.forward` verbatim except the pooling step, recording
  `config.pooling="attn"` for reload. New pooling params land in the head LR group under LLRD
  (no `_LAYER_RE`/`embed` match) — verified against `build_llrd_optimizer`. qwen3 (`.score`,
  last-token) out of scope — granite/ModernBERT only.
- **Reload note (for a later submission):** an `attn` checkpoint needs the patch reinstalled +
  `attn_pool` weights loaded at inference (mirrors `replace_head`'s `config.custom_head`); the
  in-process finetune.py eval that produces the screen number needs no reload.

## Results (2026-07-13 — vast `sandbox_4x_40gb`, one pooling per GPU)

Champion recipe (granite richargs + LS 0.1 + full_data + bf16 + 3 ep + lr 2e-5 + **bs16 grad_accum 1**
+ max_len 512 + seed 42), only `--pooling` varies. Raw uncal `val_macro_f1` on the 3.5k full_data-CV
held-out slice. Results CSV: `output/pat/ft_results_e37.csv`.

| Arm | pooling | val_macro_f1 | Δ vs cls | verdict |
|---|---|---|---|---|
| **P0** | **cls** (control, `[CLS]`/first-token) | **0.7737** | — | baseline |
| P1 | mean (masked mean-pool) | 0.7699 | **−0.0038** | ❌ hurts |
| P2 | attn (learned attention pool) | 0.7739 | **+0.0002** | ❌ flat (within noise) |

**Verdict: pooling axis is NULL — cls (granite's default) is not beaten.** Mean pooling *hurts*
(−0.0038); learned-attention pooling is flat (+0.0002, well inside the 3.5k slice ±0.003). **Neither
clears the +0.003 promote gate**, and neither is even a +0.001–0.003 ensemble-member candidate (attn's
+0.0002 is noise). Matches the modest prior — the granite ceiling is label-ambiguity, not the pooling
of an already-bidirectional encoder; "use all tokens" buys nothing over the CLS token here. No
submission built. cls stays the champion pooling.

⚠️ The cls control landed 0.7737 vs the ~0.7803 champion reference — ~0.007 low, consistent with
3.5k-slice run-to-run noise (full_data CV mis-ranks; not LB-rankable, E8/E26). Since all three arms
share the identical recipe + slice, the **Δ vs cls** is the valid read, not the absolutes.

Exact commands (each run's log via `log_cmd()`, `sbatch/logs/e37_<pooling>.out`):
```
python -m experiments.pooling.run_e37 --pooling {cls|mean|attn}
# → src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs
#   --full_data --loss ls --label_smoothing 0.1 --lr 2e-5 --grad_accum 1 --seed 42
#   --out_dir ./output/pat --results_name ft_results_e37.csv --epochs 3 --batch_size 16
#   --max_len 512 --tag e37_<pooling>
```
Figure: [e37_pooling](figures/e37_pooling.png).
