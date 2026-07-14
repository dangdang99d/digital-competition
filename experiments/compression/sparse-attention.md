# Sparse / efficient attention — shrink the attention mechanism

**Method:** reduce attention cost via a smaller sliding-window (`local_attention`), fewer
global-attention layers, or linear/sparse attention variants + recovery FT. Index:
[results.md](results.md).

## Status: 🔲 unexplored — low expected yield

granite (ModernBERT) is **already natively sparse**: local-128 sliding-window attention on
most layers, full/global attention only every 3rd layer. So the usual sparse-attention win is
largely pre-banked. Remaining knobs:
- shrink `local_attention` (128 → 64) + recovery,
- thin the global-attention layers.

Attention is only 16.7% of granite params, so this is a small **speed** lever at best. Park
unless a wall-clock profile (open question) shows attention actually dominates granite's
ms/sample on the T4.

## qwen3
Attention is 29.6% of params — the KV/q/o SVD work lives under
[low-rank-factorization.md](low-rank-factorization.md) (E45 B3), which is the more promising
attention-side lever than window-shrinking.
