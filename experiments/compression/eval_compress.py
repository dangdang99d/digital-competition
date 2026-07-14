"""Compression measurement harness — the results.md 3-metric protocol.

Evaluates a model (optionally with layers identity-dropped for zero-shot depth
probes) on the honest **3.5k held-out** (`va_eval`) of a --full_data model, and
reports, vs a reference logits file:
  1. dF1        macro-F1(this) - macro-F1(ref)
  2. flip-rate  fraction of rows whose argmax differs from ref
  3. logit delta mean/max |dlogit| + mean softmax-KL(ref || this)
Saves logits so later arms (recovery-FT models) compare against the same ref.

Usage (on a training box, repo root):
  python -m experiments.compression.eval_compress --model_dir output/..../checkpoint-16628 \
      --out output/e31a1/base.npz                        # baseline: ref for everything else
  python -m experiments.compression.eval_compress --model_dir <same> --drop_layers 4,12 \
      --out output/e31a1/zs_keep20.npz --ref output/e31a1/base.npz
  python -m experiments.compression.eval_compress --model_dir output/<recovered>/checkpoint-X \
      --out output/e31a1/rec_keep20.npz --ref output/e31a1/base.npz
"""
import argparse, os
from pathlib import Path

import numpy as np
import torch


def rebuild_va_eval(data_dir, variant, seed=42):
    """Reconstruct the --full_data 3.5k held-out (va_eval) exactly like finetune.py."""
    from sklearn.model_selection import train_test_split
    from src.data import load_samples, build_texts, split_indices, ALL_CLASSES
    samples, y = load_samples(data_dir)
    texts = build_texts(samples, input_mode="context", max_hist=None, variant=variant)
    cls2id = {c: i for i, c in enumerate(ALL_CLASSES)}
    y_ids = np.array([cls2id[a] for a in y])
    # ⚠️ int-vs-string trap (E30 lesson): finetune.py stratifies the OUTER split on the
    # STRING labels y and the INNER full_data split on int y_ids — sklearn's class
    # ordering changes row assignment, so both must match finetune byte-exactly.
    tr, va = split_indices(np.asarray(y), seed=seed)               # strings, 0.2 -> ~14k val
    va_train, va_eval = train_test_split(                          # full_data holds out 25%
        va, test_size=0.25, stratify=y_ids[va], random_state=seed)
    return [texts[i] for i in va_eval], y_ids[va_eval]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True, help="npz to save logits/y to")
    ap.add_argument("--ref", default="", help="reference logits npz (the uncompressed baseline)")
    ap.add_argument("--drop_layers", default="", help="comma idx to identity-drop (zero-shot probe)")
    ap.add_argument("--remap", default="", help="remap.npy for vocab-pruned models "
                    "(full-vocab tokenizer ids -> pruned rows; e.g. qwen3_ls zip)")
    ap.add_argument("--kept_layers", default="", help="ORIGINAL kept indices of a depth-"
                    "pruned ckpt saved before the config marker (reconstruct-load)")
    ap.add_argument("--orig_depth", type=int, default=0, help="original depth for --kept_layers")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--serialize", default="richargs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.data import macro_f1

    texts, y = rebuild_va_eval(args.data_dir, args.serialize, args.seed)
    print(f"va_eval: {len(texts)} rows (3.5k held-out, seed {args.seed})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.truncation_side = "right"

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    kept = getattr(cfg, "kept_layer_indices", None)
    if args.kept_layers:                      # override for ckpts saved before the marker
        kept = [int(x) for x in args.kept_layers.split(",")]
    factored = getattr(cfg, "factored_ffn", None)
    if factored is not None and kept is None:
        # FFN low-rank ckpt (--factor_ffn): rebuild FactoredLinear shells then load.
        # (No depth-prune here → no index-wiring issue; width prune reloads via the
        # plain path since intermediate_size is in the config.)
        from src.factored_ffn import load_factored_model
        model, missing, unexpected = load_factored_model(
            args.model_dir, torch_dtype=torch.float16, attn_implementation="eager")
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        print(f"factored-load: {factored}", flush=True)
    elif kept is not None:
        # Depth-pruned ckpt: per-layer wiring (ModernBERT global/local attn, rope theta)
        # is assigned by layer INDEX at __init__, so a plain from_pretrained of the
        # shrunken config scrambles it. Rebuild at ORIGINAL depth, prune to the kept
        # original indices (preserves each layer's wiring), then load the saved weights.
        from src.finetune import prune_layers
        orig_depth = getattr(cfg, "pruned_from_depth", None) or args.orig_depth
        assert orig_depth, "need --orig_depth for pre-marker pruned ckpts"
        cfg.num_hidden_layers = int(orig_depth)
        for k, v in (("reference_compile", False), ("_attn_implementation", "eager")):
            try: setattr(cfg, k, v)
            except Exception: pass
        model = AutoModelForSequenceClassification.from_config(cfg)
        prune_layers(model, kept)
        from safetensors.torch import load_file
        sd = load_file(str(Path(args.model_dir) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        assert all("position_ids" in m or "rotary" in m for m in missing), \
            f"missing weights: {missing[:5]}"
        model = model.half()
        print(f"reconstruct-load: depth {orig_depth} -> kept {kept}", flush=True)
    else:
        kw = dict(local_files_only=True, torch_dtype=torch.float16)
        try:    # granite/ModernBERT: avoid compiled embeddings + flash paths
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_dir, attn_implementation="eager", reference_compile=False, **kw)
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_dir, attn_implementation="eager", **kw)
    model = model.to("cuda").eval()

    if args.drop_layers:
        drop = {int(i) for i in args.drop_layers.split(",")}
        for stack_attr in ("model", "bert", "roberta"):
            enc = getattr(model, stack_attr, None)
            if enc is not None and hasattr(enc, "layers"):
                for i, l in enumerate(enc.layers):
                    if i in drop:
                        l.forward = (lambda hs, *a, **k: (hs,))
                break
        else:
            raise RuntimeError("could not find encoder layer stack to drop from")
        print(f"identity-dropped layers {sorted(drop)}", flush=True)

    remap = None
    if args.remap:
        remap = torch.from_numpy(np.load(args.remap)).long()
        pad_pruned = int(remap[tok.pad_token_id])          # pad in remapped-input space
        if model.config.pad_token_id != pad_pruned:
            # recovery ckpts saved a stale full-space pad; qwen3 last-non-pad pooling must
            # use the PRUNED-space pad the model trained with (E3 lesson). Force it.
            print(f"override config.pad_token_id {model.config.pad_token_id} -> {pad_pruned}",
                  flush=True)
            model.config.pad_token_id = pad_pruned

    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            e = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_len,
                    padding=True, return_tensors="pt")
            if remap is not None:
                e["input_ids"] = remap[e["input_ids"]]
            e = e.to("cuda")
            outs.append(model(**e).logits.float().cpu().numpy())
    logits = np.concatenate(outs, 0)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    np.savez(args.out, logits=logits, y=y)

    f1 = macro_f1(y, logits.argmax(1))
    print(f"macro-F1 = {f1:.5f}  ({args.model_dir}"
          f"{' drop=' + args.drop_layers if args.drop_layers else ''})", flush=True)

    if args.ref:
        r = np.load(args.ref)
        rl = r["logits"]
        assert rl.shape == logits.shape, "ref shape mismatch — different eval slice?"
        f1r = macro_f1(y, rl.argmax(1))
        flip = float((rl.argmax(1) != logits.argmax(1)).mean())
        d = np.abs(logits - rl)
        lp = torch.log_softmax(torch.from_numpy(logits), -1)
        rp = torch.softmax(torch.from_numpy(rl), -1)
        rlp = torch.log_softmax(torch.from_numpy(rl), -1)
        kl = float((rp * (rlp - lp)).sum(-1).mean())
        print(f"vs ref: dF1 = {f1 - f1r:+.5f} (ref {f1r:.5f})  |  "
              f"flip-rate = {flip:.4f}  |  |dlogit| mean {d.mean():.3f} max {d.max():.2f}  |  "
              f"KL(ref||this) = {kl:.5f}", flush=True)


if __name__ == "__main__":
    main()
