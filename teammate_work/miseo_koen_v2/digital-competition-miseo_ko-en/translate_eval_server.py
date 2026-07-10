# -*- coding: utf-8 -*-
# [서버/GPU] 추론-전용 번역 실험: fold1-val 전체의 한글 자연어(user content + current_prompt)만
#   영어로 번역 -> SOTA 모델로 macro-F1 원본 vs 번역 비교. 코드/파일명/args는 원문 유지.
import os, re, json, csv, copy
os.environ.pop("HF_HUB_OFFLINE",None); os.environ.pop("TRANSFORMERS_OFFLINE",None)
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROJ="/data/kimin10866/repos/데이콘"; DATA=f"{PROJ}/data"
MODEL=f"{PROJ}/results/names_2g2f_78515/models/granite-311m-v2-fold1"
BIAS=f"{PROJ}/results/names_2g2f_78515/oof/logit_bias.json"
N_SUB=int(os.environ.get("N_SUB","999999"))
ACT=["read_file","grep_search","list_directory","glob_pattern","edit_file","write_file","apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task","web_search","respond_only"]
lab2id={c:i for i,c in enumerate(ACT)}; HANGUL=re.compile(r"[가-힣]")

# ---- SOTA render (script.py와 동일) ----
def _safe(v): return "" if v is None else (v if isinstance(v,str) else str(v))
def _cj(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def _bud(t):
    try:t=int(t)
    except:return "unknown"
    return "very_low" if t<2000 else "low" if t<10000 else "medium" if t<50000 else "high"
def _el(s):
    try:s=int(s)
    except:return "unknown"
    return "early" if s<120 else "mid" if s<900 else "late"
def render_sample(sample,mh=12):
    meta=sample.get("session_meta") or {}; ws=meta.get("workspace") or {}
    hist=(sample.get("history") or [])[-mh:]; of=ws.get("open_files") or []
    lm=ws.get("language_mix") or {}; ml=max(lm.items(),key=lambda x:x[1])[0] if lm else ""
    mt=" ".join([f"tier={_safe(meta.get('user_tier'))}",f"pref={_safe(meta.get('language_pref'))}",
        f"turn={_safe(meta.get('turn_index'))}",f"budget={_bud(meta.get('budget_tokens_remaining'))}",
        f"elapsed={_el(meta.get('elapsed_session_sec'))}",f"lang={ml}",f"ci={_safe(ws.get('last_ci_status'))}",
        f"git={'dirty' if ws.get('git_dirty') else 'clean'}",f"open={len(of)}",f"loc={_safe(ws.get('loc'))}"])
    mt=mt+" openfiles="+" ".join([f.split("/")[-1] for f in of[:6]])
    parts=[]
    for it in hist:
        if it.get("role")=="user": parts.append("U: "+_safe(it.get("content")))
        elif it.get("role")=="assistant_action": parts.append(f"A[{_safe(it.get('name'))}] {_cj(it.get('args') or {})} -> {_safe(it.get('result_summary'))}")
    return " ".join(["[META]",mt,"[HIST]"," | ".join(parts),"[CUR]",_safe(sample.get("current_prompt"))])

def load_jsonl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
samples=load_jsonl(f"{DATA}/train.jsonl")
label_of={r["id"]:r["action"] for r in csv.DictReader(open(f"{DATA}/train_labels.csv",encoding="utf-8"))}
ids=[s["id"] for s in samples]; y=np.array([lab2id[label_of[i]] for i in ids])
sess=np.array([re.sub(r"-step_\d+$","",i) for i in ids])
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.zeros(len(ids)),y,groups=sess))
val=list(folds[1][1])[:N_SUB]
print(f"device={DEV} | fold1-val {len(val)}개",flush=True)

need=set()
for i in val:
    s=samples[i]
    if HANGUL.search(s.get("current_prompt") or ""): need.add(s["current_prompt"])
    for it in (s.get("history") or []):
        if it.get("role")=="user" and HANGUL.search(it.get("content") or ""): need.add(it["content"])
need=list(need); print(f"번역할 한글 문장(중복제거) {len(need)}개",flush=True)

mt_tok=MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
mtm=MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").eval().to(DEV)
tmap={}; B=64
for st in range(0,len(need),B):
    ch=need[st:st+B]
    b=mt_tok(ch,return_tensors="pt",padding=True,truncation=True,max_length=128).to(DEV)
    with torch.no_grad(): g=mtm.generate(**b,max_length=150,num_beams=1)
    for src,o in zip(ch,mt_tok.batch_decode(g,skip_special_tokens=True)): tmap[src]=o
    if st%(B*30)==0: print(f"  번역 {min(st+B,len(need))}/{len(need)}",flush=True)
del mtm; torch.cuda.empty_cache()

def tr(s):
    s2=copy.deepcopy(s)
    if s2.get("current_prompt") in tmap: s2["current_prompt"]=tmap[s2["current_prompt"]]
    for it in (s2.get("history") or []):
        if it.get("role")=="user" and it.get("content") in tmap: it["content"]=tmap[it["content"]]
    return s2

gtok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
gm=AutoModelForSequenceClassification.from_pretrained(MODEL,local_files_only=True).eval().to(DEV)
bias=np.array([json.load(open(BIAS))["bias"][c] for c in ACT],np.float32)
def predict(texts):
    out=[]
    for st in range(0,len(texts),64):
        enc=gtok(texts[st:st+64],truncation=True,max_length=512,padding=True,return_tensors="pt").to(DEV)
        with torch.no_grad(): out.append(gm(**enc).logits.float().cpu().numpy())
    return np.concatenate(out,0)

yc=y[val]
print("추론(원본)...",flush=True); lo=predict([render_sample(samples[i]) for i in val])
print("추론(번역)...",flush=True); lt=predict([render_sample(tr(samples[i])) for i in val])
fo=f1_score(yc,(lo+bias).argmax(1),average="macro"); ft=f1_score(yc,(lt+bias).argmax(1),average="macro")
lines=["="*56,f"fold1-val {len(val)}개 (SOTA=0.7676 base, held-out)",
    f"원본(한국어 유지) macro-F1={fo:.4f}",f"번역(영어 변환)   macro-F1={ft:.4f}",
    f"Δ(번역-원본) = {ft-fo:+.4f}"]
langs=np.array([(samples[i].get("session_meta") or {}).get("language_pref","?") for i in val])
for lg in ["ko","mixed","en"]:
    m=langs==lg
    if m.sum()>30: lines.append(f"  [{lg:5s} n={m.sum():5d}] 원본 {f1_score(yc[m],(lo[m]+bias).argmax(1),average='macro'):.4f} -> 번역 {f1_score(yc[m],(lt[m]+bias).argmax(1),average='macro'):.4f}")
rep="\n".join(lines); print("\n"+rep,flush=True)
open(f"{PROJ}/results/translate_eval_result.txt","w",encoding="utf-8").write(rep+"\n")
print("\n저장됨")
