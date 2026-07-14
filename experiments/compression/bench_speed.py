"""Inference speed bench: ms/sample + projected 30k wall-clock, on the real 3.5k
held-out text length distribution (batch 64, fp16, length-sorted like deployment).
Handles depth-pruned (reconstruct-load) + width-pruned (config) checkpoints.
Report the RATIO across models (transfers across HW); absolute is this GPU (3090)."""
import argparse, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from experiments.compression.eval_compress import rebuild_va_eval


def load(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    kept = getattr(cfg, "kept_layer_indices", None)
    if kept is not None:
        from src.finetune import prune_layers
        from safetensors.torch import load_file
        cfg.num_hidden_layers = int(getattr(cfg, "pruned_from_depth", 22))
        try: cfg.reference_compile = False
        except Exception: pass
        m = AutoModelForSequenceClassification.from_config(cfg, attn_implementation="sdpa")
        prune_layers(m, kept)
        m.load_state_dict(load_file(str(Path(model_dir) / "model.safetensors")), strict=False)
        m = m.half()
        tag = f"depth{len(kept)}xI{cfg.intermediate_size}"
    else:
        try:
            m = AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=torch.float16,
                attn_implementation="sdpa", reference_compile=False)
        except TypeError:
            m = AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=torch.float16,
                attn_implementation="sdpa")
        tag = f"full(L{m.config.num_hidden_layers},I{m.config.intermediate_size})"
    return m.cuda().eval(), tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--serialize", default="richargs")
    args = ap.parse_args()

    texts, _ = rebuild_va_eval("./data", args.serialize, 42)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.truncation_side = "right"
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)  # length-sort
    texts = [texts[i] for i in order]
    model, tag = load(args.model_dir)
    N = len(texts)

    def run():
        with torch.no_grad():
            for i in range(0, N, args.bs):
                e = tok(texts[i:i + args.bs], truncation=True, max_length=args.max_len,
                        padding=True, return_tensors="pt").to("cuda")
                model(**e)
    run()  # warmup
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.reps):
        t0 = time.time(); run(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    dt = min(ts)
    msps = 1000 * dt / N
    proj30k = msps * 30000 / 1000
    print(f"{tag}: {msps:.3f} ms/sample | {N} rows in {dt:.2f}s | "
          f"proj 30k = {proj30k:.1f}s ({proj30k/60:.2f} min) [3090, bs{args.bs}, sdpa fp16]",
          flush=True)


if __name__ == "__main__":
    main()
