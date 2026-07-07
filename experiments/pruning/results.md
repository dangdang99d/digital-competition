# Branch: pruning — compress the champion without losing accuracy
Git branch: `research/pruning` · Experiments: E3 (FFN whitened-SVD probe), E4 (SVD prune + recovery FT)
Baseline: Qwen3-0.6B full-FT v1@512 = 0.7682 uncal (E3: same-subset rank-1.0 row).

## Prior findings (from main-session analysis, 2026-07-07)
- KV whitened-SVD probe (training-free, 3k val): 95.5% energy @ 25% rank; −0.005 @ r=512, −0.010 @ r=256. Method works where naive width50 slice failed.
- BUT K+V = only 9.9% of params. Split: FFN 44.4%, embeddings 26.1%, attn 29.6% (Q/O 19.7%). FFN is the prize; embeddings → vocab-prune (proven recipe).
- Palu/ASVD/SVD-LLM lit: activation-aware (whitened) SVD ≫ plain; training-free always degrades; prune→recover regime (ours) can be net-zero or better.

## Results
(append: date · experiment · numbers vs baseline · [figure](figures/...) · commit)

**2026-07-07 · E3 RUNNING** — FFN whitened-SVD probe (training-free). Champion qwen3 weights
recovered from `submissions/submit_0703_qwen3_pruned.zip` (vocab-pruned but FFN/attn intact +
parity-preserving), unzipped to local disk. Extended `analysis/palu_probe.py` →
`analysis/palu_probe_ffn.py` for gate/up/down. Running on GPU 2 (97% util). Mandatory parity
self-check first: rank-1.0 row must reproduce the champion's ≈0.7734 on the 3k-val subset before
any FFN-truncation number is trusted. Curve (uncal macro-F1 vs rank ratio) + E4 go/no-go on completion.
