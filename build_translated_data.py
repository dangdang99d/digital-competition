# -*- coding: utf-8 -*-
# shard_*.json 병합 -> train.jsonl에 적용(한글 content+current_prompt만 영어로) -> data_en/ 생성.
#   코드·파일명·args·result_summary는 원문 유지. labels/sample_submission 그대로 복사.
import os, json, glob, shutil, copy
PROJ="/data/kimin10866/repos/데이콘"; DATA=f"{PROJ}/data"
TMP=f"{PROJ}/results/tmp_translate"; OUT=f"{PROJ}/data_en"; os.makedirs(OUT,exist_ok=True)

tmap={}
for f in sorted(glob.glob(f"{TMP}/shard_*.json")):
    d=json.load(open(f,encoding="utf-8")); tmap.update(d); print(f"loaded {f}: {len(d)}")
print("병합된 번역 맵:",len(tmap))

def tr_sample(s):
    s2=copy.deepcopy(s)
    if s2.get("current_prompt") in tmap: s2["current_prompt"]=tmap[s2["current_prompt"]]
    for it in (s2.get("history") or []):
        if it.get("role")=="user" and it.get("content") in tmap: it["content"]=tmap[it["content"]]
    return s2

n=0; changed=0
with open(f"{OUT}/train.jsonl","w",encoding="utf-8") as w:
    for line in open(f"{DATA}/train.jsonl",encoding="utf-8"):
        if not line.strip(): continue
        s=json.loads(line); s2=tr_sample(s)
        if json.dumps(s2,ensure_ascii=False)!=json.dumps(s,ensure_ascii=False): changed+=1
        w.write(json.dumps(s2,ensure_ascii=False)+"\n"); n+=1
print(f"train.jsonl -> data_en/train.jsonl  ({n}개, 번역적용 {changed}개)")
for fn in ["train_labels.csv","sample_submission.csv","test.jsonl"]:
    if os.path.exists(f"{DATA}/{fn}"): shutil.copy(f"{DATA}/{fn}", f"{OUT}/{fn}")
print("labels/submission/test 복사 완료 (test는 원문 그대로 — OOF 비교엔 불필요)")
