"""E37 pooling — training driver. Runs the EXACT champion recipe via src.finetune.main()
(no recipe divergence to confound the pooling axis) and injects the chosen pooling by
wrapping AutoModelForSequenceClassification.from_pretrained IN-PROCESS. Zero edits to src/.

Usage:
  python -m experiments.pooling.run_e37 --pooling cls|mean|attn [--epochs 3] [--limit N] [--tag T]

The champion recipe is fixed here: granite richargs + LS 0.1 + full_data + bf16(auto) +
epochs 3 + lr 2e-5 + bs16 + accum1 + max_len 512 + seed 42. Only --pooling varies (the axis).
Results land in output/pat/ft_..._<tag> and ft_results_e37.csv, same as any finetune run.
"""
import argparse
import sys

from loguru import logger

from experiments.pooling.pooling import install_pooling

CHAMPION = [
    "--model", "ibm-granite/granite-embedding-311m-multilingual-r2",
    "--serialize", "richargs",
    "--full_data",
    "--loss", "ls", "--label_smoothing", "0.1",
    "--lr", "2e-5",
    "--grad_accum", "1",                         # granite: NO accumulation
    "--seed", "42",
    "--out_dir", "./output/pat",
    "--results_name", "ft_results_e37.csv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", required=True, choices=["cls", "mean", "attn"])
    ap.add_argument("--epochs", default="3", help="champion=3; lower for smoke")
    ap.add_argument("--batch_size", default="16", help="champion=16 (granite real bs16, no accum); shrink for smoke")
    ap.add_argument("--max_len", default="512", help="champion=512; shrink for smoke")
    ap.add_argument("--limit", default=None, help="subsample N train rows (smoke only)")
    ap.add_argument("--tag", default=None, help="run tag; default e37_<pooling>")
    args = ap.parse_args()

    tag = args.tag or f"e37_{args.pooling}"

    # ---- inject pooling by wrapping the model loader (in-process, src/ untouched) ----
    import transformers
    _orig = transformers.AutoModelForSequenceClassification.from_pretrained

    def _wrapped(*a, **kw):
        model = _orig(*a, **kw)
        install_pooling(model, args.pooling)
        logger.info(f"E37: installed '{args.pooling}' pooling on the loaded model "
                    f"(classifier_pooling={getattr(model.config, 'classifier_pooling', '?')}, "
                    f"config.pooling={getattr(model.config, 'pooling', 'cls')})")
        return model

    transformers.AutoModelForSequenceClassification.from_pretrained = staticmethod(_wrapped)

    # ---- build the champion argv and hand off to the real finetune.main() ----
    argv = ["src.finetune"] + CHAMPION + [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_len", str(args.max_len),
        "--tag", tag,
    ]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    sys.argv = argv
    logger.info(f"E37 pooling={args.pooling} -> finetune argv: {' '.join(argv[1:])}")

    from src import finetune
    finetune.main()


if __name__ == "__main__":
    main()
