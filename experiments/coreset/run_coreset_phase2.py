"""Coreset Phase-2 orchestrator — GPUs 0 & 2 ONLY (GPUs 1 & 3 reserved for E24, user 2026-07-08).

Replaces the original 4-GPU run_coreset.py after it was stopped mid-run. Phase-1 (base +
4 cleanlab folds) is already running/done on GPUs 0 (base) and 2 (fold3); folds 0/1/2 saved
their part npzs. This runner does NOT relaunch base/folds — it:
  1. Releases GPU0 when base (its Phase-1 owner PID) exits, GPU2 when fold3 exits.
  2. Runs CPU scorers inline when their inputs land: score_dynamics (needs base's DYN npz),
     cleanlab-merge (needs all 4 oof_part npz), pvi (needs merged OOF).
  3. Runs the 10 keep-set retrains on GPUs {0,2} ONLY (CUDA_VISIBLE_DEVICES ∈ {0,2}).
GPUs 1 & 3 are NEVER used. Per-job logs sbatch/logs/coreset_<name>.out.
"""
import os
import subprocess
import time

ROOT = "/home/ocean/dacon"
os.chdir(ROOT)
PY = "/home/ocean/miniconda3/envs/dacon/bin/python"
GRANITE = "ibm-granite/granite-embedding-311m-multilingual-r2"
GPUS = [0, 2]                                   # <-- 1 & 3 excluded (E24)
OWNER = {0: 393895, 2: 448123}                 # Phase-1 job PID occupying each GPU; release when it exits
LOG, KEEP = "sbatch/logs", "experiments/coreset/keepsets"
DYN = "analysis/cache/coreset_dyn_granite.npz"
OOF = "analysis/cache/coreset_oof_granite.npz"
PARTS = [f"analysis/cache/oof_part_{i}.npz" for i in range(4)]
ENV = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE="0")
os.makedirs(LOG, exist_ok=True)
os.makedirs(KEEP, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{int(time.time()-T0):5d}s] {m}", flush=True)


def alive(pid):
    return os.path.exists(f"/proc/{pid}")


def ft(tag, extra):
    return [PY, "-m", "src.finetune", "--model", GRANITE, "--serialize", "v1", "--epochs", "3",
            "--batch_size", "4", "--grad_accum", "4", "--tag", tag, "--out_dir", "./output/pat"] + extra


def cl(args):
    return [PY, "experiments/coreset/confident_learning.py"] + args


RETRAINS = {  # name : (keep-set file, dep cpu-step)
    "cl": ("cleanlab_keep.npy", "merge"),
    "cart06": ("cart_drophard_drop06.npy", "score"), "cart15": ("cart_drophard_drop15.npy", "score"),
    "aum06": ("aum_drop06.npy", "score"), "aum15": ("aum_drop15.npy", "score"),
    "el2n06": ("el2n_drop06.npy", "score"), "el2n15": ("el2n_drop15.npy", "score"),
    "forget": ("forget_neverlearned.npy", "score"),
    "pvi06": ("pvi_drop06.npy", "pvi"), "pvi15": ("pvi_drop15.npy", "pvi"),
}
gpu = {f"rt_{n}": (ft(f"coreset_rt_{n}", ["--keep_indices", f"{KEEP}/{kf}"]), [dep])
       for n, (kf, dep) in RETRAINS.items()}

# CPU gate steps. Deps "_dyn"/"_parts" are file-existence pseudo-deps satisfied below.
cpu = {
    "score": ([PY, "experiments/coreset/score_dynamics.py", "--dyn", DYN, "--drop", "0.06", "0.15"], ["_dyn"]),
    "merge": (cl(["--merge", ",".join(PARTS), "--out", f"{KEEP}/cleanlab_keep.npy", "--oof_out", OOF]), ["_parts"]),
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


free, released, running = [], set(), {}
log(f"phase-2 start — {len(gpu)} retrains on GPUs {GPUS} (1&3 reserved for E24); "
    f"waiting on base={OWNER[0]}(GPU0) fold3={OWNER[2]}(GPU2)")

while True:
    # 0) file-based deps
    if os.path.exists(DYN):
        done.add("_dyn")
    if all(os.path.exists(p) for p in PARTS):
        done.add("_parts")
    # 1) release each GPU once its Phase-1 owner exits
    for g in GPUS:
        if g not in released and not alive(OWNER[g]):
            released.add(g); free.append(g)
            log(f"GPU{g} released (Phase-1 owner {OWNER[g]} exited)")
    # 2) run ready CPU gate steps inline
    for cn in cpu:
        if cn in started:
            continue
        if dep_ready(cpu[cn][1]) or dep_dead(cpu[cn][1]):
            started.add(cn); run_cpu(cn)
    # 3) fill free GPUs with ready retrains
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
    # 4) reap finished retrains
    for g, (n, p) in list(running.items()):
        if p.poll() is not None:
            (done if p.returncode == 0 else failed).add(n)
            log(f"GPU{g} {n} {'done' if p.returncode == 0 else 'FAILED (rc=%s)' % p.returncode}")
            del running[g]; free.append(g)
    # 5) exit when every retrain + cpu step is resolved
    if all(n in started for n in gpu) and not running and all(c in started for c in cpu):
        break
    time.sleep(15)

log(f"ALL DONE — done={sorted(n for n in done if n.startswith('rt_'))} failed={sorted(failed)}")
log("=== retrain macro-F1 (raw val) ===")
for n in [f"rt_{k}" for k in RETRAINS]:
    try:
        out = subprocess.run(["grep", "-h", "best val Macro-F1", f"{LOG}/coreset_{n}.out"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        log(f"  {n}: {out[-1].split('=')[-1].strip() if out else 'NO SCORE'}")
    except Exception as e:
        log(f"  {n}: harvest err {e}")
