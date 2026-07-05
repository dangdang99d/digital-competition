# Submission dir — bge-m3 full-FT (val 0.7349 / calibrated 0.7425)

Layout mirrors `baseline/`: `script.py` + `requirements.txt` + `model/`.
NOT in git-worthy state until `model/bge-m3/` is populated from moana:

```bash
# 1) best checkpoint -> model/bge-m3/  (checkpoint dir name may differ; pick the one that exists)
rsync -avz moana:/data/ksjang0515/dacon/output/ft_BAAI__bge-m3/checkpoint-*/ submission/model/bge-m3/
# 2) calibration file lives in the run dir, NOT the checkpoint dir
rsync -avz moana:/data/ksjang0515/dacon/output/ft_BAAI__bge-m3/logit_bias.json submission/model/bge-m3/
# 3) tokenizer files — Trainer checkpoints DON'T include them (no processing_class):
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-m3').save_pretrained('submission/model/bge-m3')"
# 4) drop training-only files from the zip payload (optimizer state is ~2x model size)
rm -f submission/model/bge-m3/optimizer.pt submission/model/bge-m3/scheduler.pt \
      submission/model/bge-m3/rng_state.pth submission/model/bge-m3/training_args.bin
```

Local smoke test (uses train.jsonl-shaped test.jsonl + sample_submission.csv):

```bash
cd submission && ln -sfn ../data data && python script.py && head output/submission.csv
```

Zip for upload (contents at zip root, like submit_0701_73.07.zip):

```bash
cd submission && zip -r ../submit_$(date +%m%d)_bgem3.zip model script.py requirements.txt
```

⚠️ `script.py`'s `serialize()` is a copy of `src/data.py serialize(max_hist=None)` —
the exact training-time text format. If training serialization ever changes, re-copy it.
⚠️ Check zip ≤ 10 GB (bge-m3 ≈ 2.3 GB fp32 safetensors — fine).
