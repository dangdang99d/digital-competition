# Unstructured pruning — zero individual weights

**Method:** zero individual weights by importance (magnitude; Wanda = |w|·‖x‖; SparseGPT =
Hessian-aware one-shot). Reaches high sparsity (50–60%+) at low accuracy cost. Index:
[results.md](results.md).

## Status: ❌ DEPRIORITIZED (user 2026-07-14) — no HW benefit

The weight matrices keep their shape, so on GPU there is **no speedup and no memory saving**
unless stored sparse and executed with sparse kernels. The only HW-executable variant is
**2:4 semi-structured** (2 of every 4 weights zero; Ampere+ sparse tensor cores ≈ up to 2×
GEMM). DACON runs T4 (Turing) — 2:4 sparse-tensor-core support is uncertain, and even where
present the framework must dispatch it.

**Decision:** pursue only as 2:4, and only if DACON's inference stack actually executes sparse
kernels — otherwise it's a paper number. All our effort goes to **structured** methods
(depth, width, vocab, low-rank), which yield unconditional dense speedup + size on any HW.

## Open question
Does the DACON T4 inference stack dispatch 2:4 sparse kernels? (Gates whether this is ever
worth revisiting.)
