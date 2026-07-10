# -*- coding: utf-8 -*-
# [서버/GPU] NLLB-200-distilled-600M(Meta 2022) 코드보존+품질 검증. whole-sentence.
import os, torch
os.environ.pop("HF_HUB_OFFLINE",None); os.environ.pop("TRANSFORMERS_OFFLINE",None)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
DEV="cuda" if torch.cuda.is_available() else "cpu"
name="facebook/nllb-200-distilled-600M"
print("loading",name,flush=True)
tok=AutoTokenizer.from_pretrained(name, src_lang="kor_Hang")
m=AutoModelForSeq2SeqLM.from_pretrained(name).eval().to(DEV)
eng=tok.convert_tokens_to_ids("eng_Latn")
tests=[
 "Pipeline 호출하는 데 어디어디 있어?",
 "UserControllerTest.java에 뭐 들어가 있나 보자",
 "getInstance 구현이 좀 이상해서 perplexity가 cross entropy exp 맞는지 확인해줘",
 "src/data/loader.py 파일 좀 열어봐",
 "혹시 이제 테스트 다시 돌려서 새 케이스까지 다 도는지 보자",
 "그 소켓 핸들러가 auth 미들웨어 타는지 db_session.commit 전에 확인해줘",
]
b=tok(tests,return_tensors="pt",padding=True).to(DEV)
with torch.no_grad(): g=m.generate(**b,forced_bos_token_id=eng,max_length=160,num_beams=1)
for t,o in zip(tests,tok.batch_decode(g,skip_special_tokens=True)):
    print("="*60,flush=True); print("원문:",t,flush=True); print("NLLB:",o,flush=True)
