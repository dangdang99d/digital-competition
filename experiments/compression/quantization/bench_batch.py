"""E46 — batch-size throughput sweep for the deployment script (length-sorted inference).

Measures rows/s + peak VRAM on the shipped model at several batch sizes, on a fixed
subset of real serialized rows. Picks the fastest batch that fits an 8 GB budget with margin.
Run AFTER the submission build (needs the GPU free).

usage: python bench_batch.py <model_dir> [n_rows]
  model_dir: fp16 dir (quantize-at-load) or a pre-quantized build dir
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

import lib_infer as L
import pkg_script as PKG

BATCHES = [64, 128, 192, 256, 384, 512]
VRAM_BUDGET_MB = 7000  # leave headroom under 8 GB (T4 has 16 GB, so this is conservative)


def load(model_dir, quant):
    import json
    cfgj = json.load(open(os.path.join(model_dir, "config.json")))
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.truncation_side = "right"
    if "quantization_config" in cfgj:
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, device_map={"": 0}).eval()
    elif quant:
        cfg = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16) if quant == "nf4"
               else BitsAndBytesConfig(load_in_8bit=True))
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, quantization_config=cfg, device_map={"": 0}).eval()
    else:
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16).cuda().eval()
    return m, tok


def main():
    os.chdir(L.REPO)
    model_dir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    quant = "nf4" if "nf4" in model_dir else ("int8" if "int8" in model_dir else None)

    _, texts, _, _ = L.build_texts()
    texts = texts[:n]
    m, tok = load(model_dir, quant)

    # warmup
    PKG.predict_logits(texts[:256], m, tok, batch_size=128)

    print(f"model={os.path.basename(model_dir.rstrip('/'))} quant={quant} n={n}\n"
          f"{'batch':>6} {'rows/s':>9} {'sec':>7} {'peakVRAM_MB':>12}")
    results = []
    for b in BATCHES:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        PKG.predict_logits(texts, m, tok, batch_size=b)
        torch.cuda.synchronize()
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e6
        rps = n / dt
        fits = vram < VRAM_BUDGET_MB
        results.append((b, rps, dt, vram, fits))
        print(f"{b:>6} {rps:>9.1f} {dt:>7.2f} {vram:>12.0f} {'' if fits else '  OVER-BUDGET'}", flush=True)

    ok = [r for r in results if r[4]]
    best = max(ok, key=lambda r: r[1]) if ok else max(results, key=lambda r: r[1])
    # project to DACON 30k test rows (this-HW throughput; T4 will differ by a HW factor)
    proj_30k = 30000 / best[1]
    print(f"\nBEST batch={best[0]} @ {best[1]:.1f} rows/s (peak {best[3]:.0f} MB)")
    print(f"  30k rows @ this HW ≈ {proj_30k:.0f}s ({proj_30k/60:.1f} min); "
          f"vs batch64 {30000/results[0][1]:.0f}s → {results[0][1]/best[1]*100-100:+.0f}% time")


if __name__ == "__main__":
    main()
