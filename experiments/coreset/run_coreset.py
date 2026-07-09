"""Coreset drop-noisy training orchestrator — GPU-slot queue over the DAG.

Keeps all GPUs busy. GPU jobs: 1 instrumented baseline (+dynamics) + 4 cleanlab OOF
folds + 10 keep-set retrains. CPU scorers (score_dynamics / cleanlab-merge / pvi) run
inline when their deps finish and unlock the retrains that depend on them:

  base ─► score_dynamics ─► retrains: cart06/15, aum06/15, el2n06/15, forget
  fold0..3 ─► merge ─► retrain: cleanlab ; merge ─► pvi ─► retrains: pvi06/15

All granite fresh from HF base, v1, 3ep. Retrain macro-F1 = each finetune log's raw
`best val Macro-F1`. Launch detached; per-job logs sbatch/logs/coreset_<name>.out.
"""
import os
import subprocess
import time

ROOT = "/home/ocean/dacon"
os.chdir(ROOT)
PY = "/home/ocean/miniconda3/envs/dacon/bin/python"
GRANITE = "ibm-granite/granite-embedding-311m-multilingual-r2"
GPUS = [0, 1, 2, 3]
LOG, KEEP = "sbatch/logs", "experiments/coreset/keepsets"
DYN = "analysis/cache/coreset_dyn_granite.npz"
OOF = "analysis/cache/coreset_oof_granite.npz"
ENV = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE="0")
os.makedirs(LOG, exist_ok=True)
os.makedirs(KEEP, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{int(time.time()-T0):5d}s] {m}", flush=True)


def ft(tag, extra):
    return [PY, "-m", "src.finetune", "--model", GRANITE, "--serialize", "v1", "--epochs", "3",
            "--batch_size", "4", "--grad_accum", "4", "--tag", tag, "--out_dir", "./output/pat"] + extra


def cl(args):
    return [PY, "experiments/coreset/confident_learning.py"] + args


PARTS = [f"analysis/cache/oof_part_{i}.npz" for i in range(4)]

# ---- GPU jobs: name -> (cmd, deps) ----
gpu = {"base": (ft("coreset_base", ["--log_dynamics", DYN]), [])}
for i in range(4):
    gpu[f"fold{i}"] = (cl(["--k", "4", "--folds", str(i), "--part_out", PARTS[i]]), [])
RETRAINS = {  # name : (keep-set file, dep cpu-step)
    "cl": ("cleanlab_keep.npy", "merge"),
    "cart06": ("cart_drophard_drop06.npy", "score"), "cart15": ("cart_drophard_drop15.npy", "score"),
    "aum06": ("aum_drop06.npy", "score"), "aum15": ("aum_drop15.npy", "score"),
    "el2n06": ("el2n_drop06.npy", "score"), "el2n15": ("el2n_drop15.npy", "score"),
    "forget": ("forget_neverlearned.npy", "score"),
    "pvi06": ("pvi_drop06.npy", "pvi"), "pvi15": ("pvi_drop15.npy", "pvi"),
}
for n, (kf, dep) in RETRAINS.items():
    gpu[f"rt_{n}"] = (ft(f"coreset_rt_{n}", ["--keep_indices", f"{KEEP}/{kf}"]), [dep])

# ---- CPU steps: name -> (cmd, deps) ----
cpu = {
    "score": ([PY, "experiments/coreset/score_dynamics.py", "--dyn", DYN, "--drop", "0.06", "0.15"], ["base"]),
    "merge": (cl(["--merge", ",".join(PARTS), "--out", f"{KEEP}/cleanlab_keep.npy", "--oof_out", OOF]),
              ["fold0", "fold1", "fold2", "fold3"]),
    "pvi": ([PY, "experiments/coreset/pvi.py", "--oof", OOF, "--drop", "0.06", "0.15"], ["merge"]),
}

done, failed, started = set(), set(), set()


def dep_ready(deps):
    return all(d in done for d in deps)


def dep_dead(deps):
    return any(d in failed for d in deps)


def run_cpu(name):
    cmd, deps = cpu[name]
    if dep_dead(deps):
        failed.add(name); log(f"CPU {name} SKIPPED (dep failed)"); return
    log(f"CPU {name} start")
    with open(f"{LOG}/coreset_{name}.out", "w") as f:
        rc = subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    (done if rc == 0 else failed).add(name)
    log(f"CPU {name} {'done' if rc == 0 else 'FAILED (rc=%d)' % rc}")


free = list(GPUS)
running = {}   # gpu -> (name, proc)
log(f"orchestrator start — {len(gpu)} GPU jobs on {len(GPUS)} GPUs")

while True:
    # 1) run any ready CPU gate steps inline (fast)
    for cn in cpu:
        if cn in started:
            continue
        if dep_ready(cpu[cn][1]) or dep_dead(cpu[cn][1]):
            started.add(cn); run_cpu(cn)
    # 2) fill free GPUs with ready GPU jobs
    for g in list(free):
        cand = None
        for n, (cmd, deps) in gpu.items():
            if n in started:
                continue
            if dep_dead(deps):
                failed.add(n); started.add(n); log(f"GPU {n} SKIPPED (dep failed)"); continue
            if dep_ready(deps):
                cand = (n, cmd); break
        if not cand:
            continue
        n, cmd = cand
        started.add(n)
        e = dict(ENV, CUDA_VISIBLE_DEVICES=str(g))
        fh = open(f"{LOG}/coreset_{n}.out", "w")
        running[g] = (n, subprocess.Popen(cmd, env=e, stdout=fh, stderr=subprocess.STDOUT))
        free.remove(g)
        log(f"GPU{g} launch {n}")
    # 3) reap finished GPU jobs
    for g, (n, p) in list(running.items()):
        if p.poll() is not None:
            (done if p.returncode == 0 else failed).add(n)
            log(f"GPU{g} {n} {'done' if p.returncode == 0 else 'FAILED (rc=%s)' % p.returncode}")
            del running[g]; free.append(g)
    # 4) exit when everything is resolved
    if len(started) >= len(gpu) and not running and all(c in started for c in cpu):
        break
    time.sleep(15)

log(f"ALL DONE — done={len(done)} failed={sorted(failed)}")
# harvest retrain macro-F1
log("=== retrain macro-F1 (raw val) ===")
for n in ["base"] + [f"rt_{k}" for k in RETRAINS]:
    try:
        out = subprocess.run(["grep", "-h", "best val Macro-F1", f"{LOG}/coreset_{n}.out"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        log(f"  {n}: {out[-1].split('=')[-1].strip() if out else 'NO SCORE'}")
    except Exception as e:
        log(f"  {n}: harvest err {e}")
