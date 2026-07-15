"""Quick vLLM throughput + output diagnostic for the E39 LLM generation slowness."""
import time
from vllm import LLM, SamplingParams
from experiments.word_aug.gen_llm import SYS, build_prompt

SAMPLES = [
    ["make sure the whole test suite still passes after that sweep",
     "이 값으로 config 로드만 한번 태워서 파싱 에러 없는지 확인해줘 대충 말고"],
    ["spin up the dev server real quick so i can eyeball it",
     "필드명 바꿨으면 마이그레이션 만들어야지"],
    ["add the handler404 = ... line pointing at a custom view"],
    ["rayon 쓰는 게 깔끔할듯. 최신 권장 사용법 좀 찾아봐 급해"],
]

llm = LLM(model="Qwen/Qwen2.5-14B-Instruct-AWQ", quantization="awq_marlin", dtype="float16",
          max_model_len=2048, gpu_memory_utilization=0.90)
mc = llm.llm_engine.model_config
print(f"QUANT={mc.quantization}  DTYPE={mc.dtype}", flush=True)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

prompts = []
for texts in SAMPLES * 4:            # 16 prompts
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": build_prompt(texts)}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

t = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"\n{len(outs)} prompts in {dt:.1f}s | {ntok} out tok | {ntok/dt:.1f} tok/s | "
      f"{dt/len(outs):.2f}s/prompt | avg {ntok/len(outs):.0f} tok/out", flush=True)
for o in outs[:4]:
    print("OUT:", repr(o.outputs[0].text[:220]), flush=True)
