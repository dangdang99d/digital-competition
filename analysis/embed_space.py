"""Extract pooled [CLS] embeddings for ALL train samples with a fine-tuned
checkpoint, then project to 2D with UMAP and t-SNE (PCA-50 preprocessing).

Outputs (out_dir):
  embeddings_<tag>.npy   (N, hidden) float16 CLS embeddings
  proj_<tag>.npz         umap (N,2), tsne (N,2), y (N,) class ids

Usage:
  python -m analysis.embed_space --ckpt output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500
  python -m analysis.embed_space --skip_extract   # reuse saved embeddings
"""
import argparse
import os

import numpy as np
import torch
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples


def extract(ckpt, texts, max_len, batch, out_path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # minimize padding
    embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            cls = h[:, 0].float().cpu().numpy().astype(np.float16)
            for j, i in enumerate(idx):
                embs[i] = cls[j]
            if (s // batch) % 100 == 0:
                logger.info(f"embedded {s + len(idx)}/{len(texts)}")
    np.save(out_path, embs)
    logger.success(f"embeddings {embs.shape} -> {out_path}")
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="hist0")
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--skip_extract", action="store_true")
    args = ap.parse_args()

    samples, y = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    emb_path = os.path.join(args.out_dir, f"embeddings_{args.tag}.npy")

    if args.skip_extract and os.path.exists(emb_path):
        embs = np.load(emb_path)
        logger.info(f"reusing {emb_path} {embs.shape}")
    else:
        texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
        embs = extract(args.ckpt, texts, args.max_len, args.batch, emb_path)

    x = embs.astype(np.float32)
    from sklearn.decomposition import PCA
    x50 = PCA(n_components=50, random_state=42).fit_transform(x)
    logger.info(f"PCA50 explained var: {None}")  # cheap log anchor

    import umap
    logger.info("UMAP...")
    u = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                  random_state=42, verbose=True).fit_transform(x)

    from sklearn.manifold import TSNE
    logger.info("t-SNE (PCA-50 input)...")
    t = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
             random_state=42, verbose=2, n_jobs=-1).fit_transform(x50)

    out = os.path.join(args.out_dir, f"proj_{args.tag}.npz")
    np.savez_compressed(out, umap=u.astype(np.float32), tsne=t.astype(np.float32),
                        y=y_ids)
    logger.success(f"projections -> {out}")


if __name__ == "__main__":
    main()
