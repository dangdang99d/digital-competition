# -*- coding: utf-8 -*-
# [서버/GPU] NLLB-600M로 train.jsonl 한글 자연어(content+current_prompt) 유니크 문자열 NSHARD 등분 번역.
#   whole-sentence (NLLB가 코드/경로/영어 잘 보존). SHARD 담당분만 -> results/tmp_translate/shard_{SHARD}.json
import os, re, json
os.environ.pop("HF_HUB_OFFLINE",None); os.environ.pop("TRANSFORMERS_OFFLINE",None)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROJ="/data/kimin10866/repos/데이콘"; DATA=f"{PROJ}/data"
OUT=f"{PROJ}/results/tmp_translate"; os.makedirs(OUT,exist_ok=True)
SHARD=int(os.environ["SHARD"]); NSHARD=int(os.environ.get("NSHARD","4"))
HANGUL=re.compile(r"[가-힣]")
DEV="cuda" if torch.cuda.is_available() else "cpu"

def load_jsonl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
samples=load_jsonl(f"{DATA}/train.jsonl")
uniq=set()
for s in samples:
    if HANGUL.search(s.get("current_prompt") or ""): uniq.add(s["current_prompt"])
    for it in (s.get("history") or []):
        if it.get("role")=="user" and HANGUL.search(it.get("content") or ""): uniq.add(it["content"])
allk=sorted(uniq); mine=allk[SHARD::NSHARD]
print(f"[shard {SHARD}/{NSHARD}] 유니크 {len(allk)} 중 담당 {len(mine)} | dev={DEV}",flush=True)

NAME="facebook/nllb-200-distilled-600M"
tok=AutoTokenizer.from_pretrained(NAME, src_lang="kor_Hang")
m=AutoModelForSeq2SeqLM.from_pretrained(NAME).eval().to(DEV)
ENG=tok.convert_tokens_to_ids("eng_Latn")
res={}; B=48
for st in range(0,len(mine),B):
    ch=mine[st:st+B]
    b=tok(ch,return_tensors="pt",padding=True,truncation=True,max_length=160).to(DEV)
    with torch.no_grad(): g=m.generate(**b,forced_bos_token_id=ENG,max_length=180,num_beams=1)
    for src,o in zip(ch,tok.batch_decode(g,skip_special_tokens=True)): res[src]=o
    if st%(B*40)==0: print(f"  [shard {SHARD}] {min(st+B,len(mine))}/{len(mine)}",flush=True)
json.dump(res,open(f"{OUT}/shard_{SHARD}.json","w",encoding="utf-8"),ensure_ascii=False)
print(f"[shard {SHARD}] DONE -> {OUT}/shard_{SHARD}.json ({len(res)})",flush=True)
