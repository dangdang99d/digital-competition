"""E15a QE — reference-free quality on the saved (orig KO -> NLLB EN) pairs.

Primary proxy: cross-lingual cosine similarity between the Korean source and the
English MT, using BAAI/bge-m3 (cached, strong multilingual dense embedder). High
cosine == the meaning survived translation. This is the LaBSE/LASER-style
reference-free QE approach (no reference translation needed).

Bonus: Unbabel CometKiwi (wmt22-cometkiwi-da) if `comet` imports cleanly.
"""
import json, os, sys, statistics

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

OUT = "experiments/translation"


def bge_embed(texts, tok, model, dev, bs=32):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = texts[i:i + bs]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(dev)
            out = model(**enc)
            e = out.last_hidden_state[:, 0]          # bge-m3 dense = CLS token
            e = F.normalize(e, p=2, dim=1)
            embs.append(e.float().cpu())
    return torch.cat(embs, 0)


def main():
    pairs = json.load(open(os.path.join(OUT, "e15a_pairs.json"), encoding="utf-8"))
    src = [p["orig"] for p in pairs]
    mt = [p["trans"] for p in pairs]
    dev = "cuda"

    print("loading bge-m3 for cross-lingual cosine QE", flush=True)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).to(dev).eval()
    es = bge_embed(src, tok, model, dev)
    em = bge_embed(mt, tok, model, dev)
    cos = (es * em).sum(1)                            # already normalized
    cos_list = cos.tolist()
    xling_mean = statistics.mean(cos_list)
    xling_med = statistics.median(cos_list)
    # fraction below a "meaning likely lost" threshold
    low = sum(1 for c in cos_list if c < 0.55) / len(cos_list)

    del model
    torch.cuda.empty_cache()

    # ---- optional CometKiwi ----
    comet_mean = None
    comet_note = "not attempted"
    try:
        from comet import download_model, load_from_checkpoint
        ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
        cmodel = load_from_checkpoint(ckpt)
        data = [{"src": s, "mt": t} for s, t in zip(src, mt)]
        res = cmodel.predict(data, batch_size=16, gpus=1)
        comet_mean = float(statistics.mean(res["scores"]))
        comet_note = "Unbabel/wmt22-cometkiwi-da"
    except Exception as e:
        comet_note = f"skipped: {type(e).__name__}: {str(e)[:120]}"

    qe = {
        "n_pairs": len(pairs),
        "xling_cosine_bge_m3_mean": xling_mean,
        "xling_cosine_bge_m3_median": xling_med,
        "frac_cosine_below_0.55": low,
        "comet_kiwi_mean": comet_mean,
        "comet_note": comet_note,
    }
    # attach lowest-cosine pairs for eyeballing
    order = sorted(range(len(pairs)), key=lambda i: cos_list[i])
    worst = [{"cos": round(cos_list[i], 3), "orig": pairs[i]["orig"][:180],
              "trans": pairs[i]["trans"][:180]} for i in order[:12]]

    json.dump({"qe": qe, "worst_cosine_pairs": worst},
              open(os.path.join(OUT, "e15a_qe.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n==== E15a QE ====", flush=True)
    for k, v in qe.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
