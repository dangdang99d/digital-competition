"""E36 arm A (decisive) — multi-view averaging over EQUAL-QUALITY close-cousin views.
Champion e8a richargs model; views richargs/richfiles/richmeta (E25: all ~0.780, diverse
framing). Val is full_data-contaminated (absolute F1 inflated) but the near-equal views are
memorized comparably, so the RELATIVE 'does averaging beat single view' signal is clean.
Also probes richargs history-truncation (h_full vs h12) for a 2nd diversity axis."""
import numpy as np, torch
from loguru import logger
from src.data import CLASS_TO_ID, build_texts, load_samples, macro_f1, split_indices

CKPT="output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e8a_ls_richargs_full/checkpoint-8314"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok=AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForSequenceClassification.from_pretrained(CKPT, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
model.config.pad_token_id=tok.pad_token_id
samples,y=load_samples("./data"); _,va=split_indices(y,seed=42)
vs=[samples[i] for i in va]; yt=np.array([CLASS_TO_ID[y[i]] for i in va])
def fwd(texts,bs=32):
    order=np.argsort([len(t) for t in texts]); out=np.zeros((len(texts),model.config.num_labels),np.float32)
    with torch.no_grad():
        for i in range(0,len(order),bs):
            idx=order[i:i+bs]; enc=tok([texts[j] for j in idx],truncation=True,max_length=512,padding=True,return_tensors="pt").to("cuda")
            out[idx]=model(**enc).logits.float().cpu().numpy()
    return out
def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
views={"richargs_full":dict(variant="richargs"),"richfiles_full":dict(variant="richfiles"),
       "richmeta_full":dict(variant="richmeta"),"richargs_h12":dict(variant="richargs",max_hist=12)}
P={}
for n,kw in views.items():
    P[n]=sm(fwd(build_texts(vs,**kw))); logger.info(f"[single] {n:16s} F1={macro_f1(yt,P[n].argmax(1)):.4f}")
base=macro_f1(yt,P["richargs_full"].argmax(1)); logger.info(f"BASELINE richargs_full {base:.4f}")
def avg(ns): return macro_f1(yt,np.mean([P[n] for n in ns],0).argmax(1))
for lbl,ns in {"avg[argfull,filesfull,metafull]":["richargs_full","richfiles_full","richmeta_full"],
               "avg[argfull,filesfull]":["richargs_full","richfiles_full"],
               "avg[argfull,metafull]":["richargs_full","richmeta_full"],
               "avg[argfull,argh12]":["richargs_full","richargs_h12"],
               "avg[all4]":list(views)}.items():
    f=avg(ns); logger.info(f"[avg] {lbl:34s} F1={f:.4f} (Δ{f-base:+.4f}){'  <==BEATS' if f>base else ''}")
