# Token reduction (structured, sequence) — drop/merge tokens mid-network

**Method:** LTP (Learned Token Pruning) learns per-layer attention-score thresholds; tokens
below threshold are dropped as the sequence flows through — classification only needs the
pooled vector. ToMe (token merging) is the fallback. Pure **speed** lever (~2× lit. claim),
orthogonal to depth/width; compounds with them and with E17 batching. Index:
[results.md](results.md).

## Status: 🕐 DEFERRED (user 2026-07-08, E18)
Parked — the LTP work as originally scoped "is not what was intended"; awaits a fresh spec.
WIP code (`src/ltp_*`, `--ltp_*` flags) is default-off and isolated. **Do NOT dispatch**
until re-specced.

## Notes
- Sources: *Learned Token Pruning for Transformers* (Kim et al., KDD 2022); PoWER-BERT
  lineage (Goyal et al., ICML 2020).
- Reserved as axis A4 (granite E31) / B4 (qwen3 E45) in the combination search, but the slot
  stays empty until the user re-specs it.
- Revive if the speed score binds after the depth/width/quant stack is chosen.
