"""Coreset Phase-2b — add GPU 3 to the retrain pool (user freed GPU3 2026-07-09 ~00:20).

GPUs 0, 2, 3 (GPU 1 stays reserved — E24 is now RUNNING on it). Supersedes run_coreset_phase2.py
(GPUs 0,2), which is killed. Two retrains are already in flight and get orphaned (they keep running):
  rt_cart06 pid 488495 on GPU0, rt_cl pid 495518 on GPU2.
This runner: GPU3 is free immediately; GPU0/GPU2 release when cart06/cl exit. It runs the 8 REMAINING
retrains (all keep-sets already on disk, so no scorers/CPU steps) across {0,2,3}, waits for the two
orphaned ones, then harvests all 10 raw macro-F1s. GPU 1 is NEVER used.
"""
import os
import subprocess
import time

ROOT = "/home/ocean/dacon"
os.chdir(ROOT)
PY = "/home/ocean/miniconda3/envs/dacon/bin/python"
GRANITE = "ibm-granite/granite-embedding-311m-multilingual-r2"
GPUS = [0, 2, 3]                               # GPU 1 excluded — E24 running there
OWNER = {0: 488495, 2: 495518}                 # in-flight retrains; GPU free when owner exits (GPU3 = no owner = free now)
LOG, KEEP = "sbatch/logs", "experiments/coreset/keepsets"
ENV = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE="0")
T0 = time.time()


def log(m):
    print(f"[{int(time.time()-T0):5d}s] {m}", flush=True)


def alive(pid):
    return os.path.exists(f"/proc/{pid}")


def ft(tag, extra):
    return [PY, "-m", "src.finetune", "--model", GRANITE, "--serialize", "v1", "--epochs", "3",
            "--batch_size", "4", "--grad_accum", "4", "--tag", tag, "--out_dir", "./output/pat"] + extra


KEEPSET = {  # name -> keep-set file  (all 10; cart06/cl are the orphaned in-flight ones)
    "cl": "cleanlab_keep.npy",
    "cart06": "cart_drophard_drop06.npy", "cart15": "cart_drophard_drop15.npy",
    "aum06": "aum_drop06.npy", "aum15": "aum_drop15.npy",
    "el2n06": "el2n_drop06.npy", "el2n15": "el2n_drop15.npy",
    "forget": "forget_neverlearned.npy",
    "pvi06": "pvi_drop06.npy", "pvi15": "pvi_drop15.npy",
}
REMAINING = [n for n in KEEPSET if n not in ("cart06", "cl")]   # 8 still to run
gpu = {f"rt_{n}": ft(f"coreset_rt_{n}", ["--keep_indices", f"{KEEP}/{KEEPSET[n]}"]) for n in REMAINING}

done, failed, started = set(), set(), set()
free, released, running = [], set(), {}
log(f"phase-2b start — {len(gpu)} remaining retrains on GPUs {GPUS} (GPU1 = E24, untouched); "
    f"orphaned in-flight: rt_cart06({OWNER[0]}/GPU0) rt_cl({OWNER[2]}/GPU2)")

while True:
    # release each GPU: GPU3 immediately (no owner), GPU0/2 when their in-flight retrain exits
    for g in GPUS:
        if g in released:
            continue
        owner = OWNER.get(g)
        if owner is None or not alive(owner):
            released.add(g); free.append(g)
            log(f"GPU{g} free" + (f" (owner {owner} exited)" if owner else " (no owner)"))
    # fill free GPUs with remaining retrains (all deps already satisfied — keep-sets on disk)
    for g in list(free):
        nxt = next((n for n in gpu if n not in started), None)
        if nxt is None:
            break
        started.add(nxt)
        e = dict(ENV, CUDA_VISIBLE_DEVICES=str(g))
        fh = open(f"{LOG}/coreset_{nxt}.out", "w")
        running[g] = (nxt, subprocess.Popen(gpu[nxt], env=e, stdout=fh, stderr=subprocess.STDOUT))
        free.remove(g)
        log(f"GPU{g} launch {nxt}")
    # reap
    for g, (n, p) in list(running.items()):
        if p.poll() is not None:
            (done if p.returncode == 0 else failed).add(n)
            log(f"GPU{g} {n} {'done' if p.returncode == 0 else 'FAILED (rc=%s)' % p.returncode}")
            del running[g]; free.append(g)
    if all(n in started for n in gpu) and not running:
        break
    time.sleep(15)

# wait for the two orphaned in-flight retrains before harvesting
while any(alive(pid) for pid in OWNER.values()):
    log("waiting on orphaned in-flight retrains (cart06/cl) to finish...")
    time.sleep(20)

log(f"ALL DONE — remaining done={sorted(done)} failed={sorted(failed)}")
log("=== retrain macro-F1 (raw val) — all 10 ===")
for n in KEEPSET:
    try:
        out = subprocess.run(["grep", "-h", "best val Macro-F1", f"{LOG}/coreset_rt_{n}.out"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        log(f"  rt_{n}: {out[-1].split('=')[-1].strip() if out else 'NO SCORE'}")
    except Exception as e:
        log(f"  rt_{n}: harvest err {e}")
