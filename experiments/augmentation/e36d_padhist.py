"""E36 arm A probe (user idea) — FORCE history length to 12 by padding.
Tests length-bias hypothesis: does padding short histories to 12 (duplicate/random)
help, or is the monotonic depth curve pure information content?
Padding goes at the FRONT so real recent events stay adjacent to the prompt.
Honest model e9 (v1, clean 14k val)."""
import copy, numpy as np, torch
from loguru import logger
from src.data import CLASS_TO_ID, build_texts, load_samples, macro_f1, split_indices
CKPT="output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e9_granite_ls/checkpoint-10500"
TGT=12; rng=np.random.default_rng(0)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok=AutoTokenizer.from_pretrained(CKPT,trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForSequenceClassification.from_pretrained(CKPT,torch_dtype=torch.float16,trust_remote_code=True).cuda().eval()
model.config.pad_token_id=tok.pad_token_id
samples,y=load_samples("./data"); _,va=split_indices(y,seed=42)
vs=[samples[i] for i in va]; yt=np.array([CLASS_TO_ID[y[i]] for i in va])
H=np.array([len(s["history"]) for s in vs])
pool=[t for s in vs for t in s["history"]]  # global event pool for random padding
def pad(sample, mode):
    h=sample["history"]
    if len(h)>=TGT or len(h)==0:  # nothing to duplicate for empty; 12 already full
        return sample
    need=TGT-len(h)
    if mode=="tile":       front=(h*((need//len(h))+1))[:need]
    elif mode=="repfirst": front=[h[0]]*need
    elif mode=="random":   front=[pool[k] for k in rng.integers(0,len(pool),need)]
    s2=copy.copy(sample); s2["history"]=list(front)+list(h); return s2
def fwd(texts,bs=32):
    order=np.argsort([len(t) for t in texts]); out=np.zeros((len(texts),model.config.num_labels),np.float32)
    with torch.no_grad():
        for i in range(0,len(order),bs):
            idx=order[i:i+bs]; enc=tok([texts[j] for j in idx],truncation=True,max_length=512,padding=True,return_tensors="pt").to("cuda")
            out[idx]=model(**enc).logits.float().cpu().numpy()
    return out
def evalmode(mode):
    ss=[pad(s,mode) for s in vs] if mode!="natural" else vs
    pred=fwd(build_texts(ss,variant="v1")).argmax(1)
    short=H<TGT  # rows the padding actually changes
    return macro_f1(yt,pred), macro_f1(yt[short],pred[short]), macro_f1(yt[~short],pred[~short])
base=evalmode("natural"); logger.info(f"natural   overall {base[0]:.4f} | short(H<12) {base[1]:.4f} | full(H=12) {base[2]:.4f}")
for m in ["tile","repfirst","random"]:
    r=evalmode(m); logger.info(f"pad-{m:8s} overall {r[0]:.4f} (Δ{r[0]-base[0]:+.4f}) | short {r[1]:.4f} (Δ{r[1]-base[1]:+.4f}) | full {r[2]:.4f}")
logger.info(f"[note] {(H<TGT).sum()}/{len(H)} rows have <12 real events (padding acts on these)")
