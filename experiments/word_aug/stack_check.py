"""E39 arm B3 (Qwen local) — pre-flight stack check. NO model load; runs in seconds.

Verifies the box can serve FP8 on its GPUs BEFORE we spend minutes loading 30GB. This is the
first gate that guards against the previous failure mode (AWQ garbage on an unsupported kernel):
here we confirm compute capability + torch/CUDA + vLLM versions are FP8-capable. The real output
sanity check is the `--smoke` gate in gen_llm_vllm.py (which does load the model).

  python -m experiments.word_aug.stack_check
Exit 0 = stack looks FP8-capable; exit 1 = something is off (read the message).
"""
import sys


def main():
    ok = True
    try:
        import torch
    except Exception as e:
        print(f"FAIL: torch import: {e}")
        sys.exit(1)
    print(f"torch      {torch.__version__}  (CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() == False")
        sys.exit(1)
    n = torch.cuda.device_count()
    print(f"GPUs       {n}")
    caps = set()
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        caps.add(cap)
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU{i}: {name}  sm_{cap[0]}{cap[1]}  {total/1e9:.1f}GB ({free/1e9:.1f} free)")
    # FP8 needs Ada sm_89 / Hopper sm_90 / Blackwell sm_120+. Below that -> no hardware FP8.
    minc = min(caps)
    if minc[0] < 9 and not (minc[0] == 8 and minc[1] == 9):
        print(f"FAIL: min compute capability sm_{minc[0]}{minc[1]} has NO hardware FP8 "
              f"(need Ada sm_89 / Hopper sm_90 / Blackwell sm_120). Use bf16 fallback instead.")
        ok = False
    else:
        print(f"OK: sm_{minc[0]}{minc[1]} supports hardware FP8")
    try:
        import vllm
        print(f"vllm       {vllm.__version__}")
        # Blackwell (sm_120) FP8 needs a recent vLLM; warn loudly if old on Blackwell.
        if minc[0] >= 12:
            maj_min = tuple(int(x) for x in vllm.__version__.split(".")[:2])
            if maj_min < (0, 6):
                print(f"WARN: vLLM {vllm.__version__} may predate solid Blackwell(sm_120) FP8 "
                      f"support — if smoke fails, upgrade vLLM or run --dtype bfloat16.")
    except Exception as e:
        print(f"FAIL: vllm import: {e}")
        ok = False
    print("STACK_OK" if ok else "STACK_BAD")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
