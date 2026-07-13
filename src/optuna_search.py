"""E28: optuna hyperparameter search around the champion recipe (granite+richargs+LS).

One WORKER process per GPU; all workers share one SQLite study, so run e.g.:

    CUDA_VISIBLE_DEVICES=0 python -u -m src.optuna_search --gpu_tag g0 --n_trials 30 &
    CUDA_VISIBLE_DEVICES=1 python -u -m src.optuna_search --gpu_tag g1 --n_trials 30 &

Each trial launches src.finetune as a SUBPROCESS (crash isolation; the run's own
log_cmd/self-logging conventions apply) on the session-grouped fold-0 split
(leakage-free protocol — the row-stratified/full_data slices are known to mis-rank,
E8). The worker streams the subprocess log, reports per-epoch eval_macro_f1 to the
pruner, and kills losing runs early. Weights are NOT kept (search reruns are cheap;
promotion retrains use --all_data anyway).

Fixed, not searched (settled by prior experiments): granite backbone, richargs,
--loss ls, --init_seed 42 (soup/ensemble-compatible), max_len 512, bf16-auto.
"""
import argparse
import os
import re
import signal
import subprocess
import sys

import optuna
from loguru import logger

from src.runlog import log_cmd

MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# per-epoch eval line printed by HF Trainer, and finetune.py's final success line
RE_EPOCH_F1 = re.compile(r"'eval_macro_f1': ([0-9.]+)")
RE_FINAL_F1 = re.compile(r"best val Macro-F1 = ([0-9.]+)")


def cuda_alive():
    """cuInit ground truth — nvidia-smi lies after a bus-falloff (fleet memory 07-11)."""
    import ctypes
    try:
        return ctypes.CDLL("libcuda.so.1").cuInit(0) == 0
    except OSError:
        return False


def build_cmd(trial, args):
    """Suggest hyperparameters and assemble the finetune.py invocation."""
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    # epochs: FIXED for the AWP search (E35), still searched for E28 back-compat.
    # We RAM-snapshot the BEST-epoch checkpoint, which makes "more epochs" a free lunch
    # in the objective (more chances at a high best epoch) → a *searched* epochs drifts
    # to 5 by noise (observed in E28) without being genuinely better, and wastes compute.
    # Fixing epochs + best-epoch selection extracts each config's true peak; 4 contains
    # it (E28 curves peak at ep3; AWP regularizes so its peak may sit a touch later).
    epochs = 4 if args.search_awp else trial.suggest_int("epochs", 2, 5)
    eff_batch = trial.suggest_categorical("eff_batch", [8, 16, 32])
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.15)
    ls_eps = trial.suggest_float("label_smoothing", 0.02, 0.20)
    wd = trial.suggest_float("weight_decay", 1e-3, 0.1, log=True)
    # E19 fast shape: real batches up to the measured-safe bs16, accumulation only
    # for the eff-32 arm (bs32@512 unmeasured on 24GB -> bs16xga2, identical grads).
    # grad_accum pinned explicitly (finetune.py default is 4!); grad-ckpt auto-off.
    bs = min(eff_batch, 16)
    cmd = [PY, "-u", "-m", "src.finetune",
           "--model", MODEL, "--serialize", "richargs", "--loss", "ls",
           "--init_seed", "42", "--group_by_length",
           "--session_fold", str(args.fold), "--session_splits", "5",
           "--lr", f"{lr:.3e}", "--epochs", str(epochs),
           "--batch_size", str(bs), "--grad_accum", str(eff_batch // bs),
           "--warmup_ratio", f"{warmup:.4f}", "--label_smoothing", f"{ls_eps:.4f}",
           "--weight_decay", f"{wd:.4e}",
           "--tag", f"{args.exp}_t{trial.number:03d}",
           "--out_dir", args.out_dir,
           "--results_name", f"optuna_{args.gpu_tag}.csv"]  # per-worker CSV: no append race
    # E35: joint AWP-knob search on top of the recipe search. AWP was LB-validated
    # (0.78557, single-model SOTA) at the blind defaults gamma=1e-3/adv_lr=1e-4/
    # start_epoch=1.0 (E34) — that config is enqueued as the study anchor. start_epoch
    # is capped at {0,1} so AWP is active for the LATER epochs the pruner reports on
    # (it fires only from state.epoch>=start_epoch → epoch 2 hits AWP for BOTH values,
    # keeping MedianPruner meaningful; start_epoch=2 would leave 2 pre-AWP epochs that
    # look identical to a plain run and mis-prune).
    if args.search_awp:
        awp_gamma = trial.suggest_float("awp_gamma", 5e-4, 5e-3, log=True)
        awp_lr = trial.suggest_float("awp_lr", 3e-5, 3e-4, log=True)
        awp_start = trial.suggest_categorical("awp_start_epoch", [0, 1])
        cmd += ["--awp_gamma", f"{awp_gamma:.4e}", "--awp_lr", f"{awp_lr:.4e}",
                "--awp_start_epoch", f"{float(awp_start):.1f}"]
    return cmd, epochs


def run_trial(trial, args):
    cmd, n_epochs = build_cmd(trial, args)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", f"trial_{trial.number:03d}.log")
    logger.info(f"trial {trial.number}: {' '.join(cmd[4:])}")
    final_f1, epoch_seen = None, 0
    # own process group so a prune can kill finetune + any children in one shot
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            preexec_fn=os.setsid)
    try:
        with open(log_path, "w") as lf:
            for line in proc.stdout:
                lf.write(line)
                m = RE_EPOCH_F1.search(line)
                if m:
                    epoch_seen += 1
                    trial.report(float(m.group(1)), step=epoch_seen)
                    if epoch_seen < n_epochs and trial.should_prune():
                        raise optuna.TrialPruned(
                            f"pruned at epoch {epoch_seen}/{n_epochs}")
                m = RE_FINAL_F1.search(line)
                if m:
                    final_f1 = float(m.group(1))
        rc = proc.wait()
    finally:
        if proc.poll() is None:                       # still running (prune/ctrl-c)
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
    if final_f1 is None:
        raise RuntimeError(f"trial {trial.number}: no final F1 in log "
                           f"(exit {rc}) — see {log_path}")
    # keep weights ONLY for the running best (user 2026-07-12: retrieve weights too);
    # losers are deleted — promotion retrains --all_data from scratch anyway
    try:
        keep = final_f1 >= trial.study.best_value   # best_value excludes this trial
    except ValueError:                              # no completed trials yet
        keep = True
    run_dir = os.path.join(args.out_dir,
                           f"ft_{MODEL.replace('/', '__')}_{args.exp}_t{trial.number:03d}")
    if keep:
        logger.info(f"trial {trial.number}: new best {final_f1:.4f} — weights kept")
    elif os.path.isdir(run_dir):
        import shutil
        shutil.rmtree(run_dir)
    return final_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_tag", required=True,
                    help="short worker name (e.g. g0/g1) — used in per-worker CSV "
                         "names; set the actual GPU via CUDA_VISIBLE_DEVICES")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--fold", type=int, default=0,
                    help="session-grouped fold used as the search holdout")
    ap.add_argument("--study", default="e28_granite_ls")
    ap.add_argument("--db", default="output/optuna/e28.db")
    ap.add_argument("--journal", default="",
                    help="MULTI-BOX: path to a JournalStorage file on SHARED NFS "
                         "(e.g. /nfs/dacon/output/optuna/e35_journal.log). When set, "
                         "overrides --db (SQLite) — SQLite over NFS corrupts under "
                         "cross-host locking; JournalStorage w/ symlink lock is NFS-safe. "
                         "Point every worker on every box at the SAME path to share one "
                         "distributed study.")
    ap.add_argument("--out_dir", default="output/optuna/trials")
    ap.add_argument("--exp", default="e28",
                    help="tag prefix for trial runs/dirs (e28 recipe search, "
                         "e35 AWP joint search) — keeps studies from colliding")
    ap.add_argument("--search_awp", action="store_true",
                    help="E35: also search AWP knobs (gamma/adv_lr/start_epoch) "
                         "jointly with the recipe, on top of the champion anchor")
    ap.add_argument("--enqueue_anchor", action="store_true",
                    help="E35: enqueue the LB-validated config (champion recipe + AWP "
                         "gamma1e-3/lr1e-4/start1 = LB 0.78557) as the first trial so "
                         "every result reads as a Delta vs a known-LB point. Pass on "
                         "ONE worker only (avoids duplicate anchors)")
    args = ap.parse_args()
    log_cmd()

    assert cuda_alive(), \
        "cuInit failed — host driver is poisoned (see fleet memory); do not start trials"

    # Pruner: E28 used MedianPruner (kill trials below the median epoch-curve). For the
    # AWP search (E35) we DISABLE pruning (NopPruner): AWP's benefit appears only in the
    # LATER epochs (it activates from start_epoch), so early-epoch pruning would risk
    # killing a slow-start-but-blooms config — and with epochs fixed at 4 on 16x5090,
    # the compute saved by pruning is marginal vs that downside. Run every trial full.
    if args.search_awp:
        pruner = optuna.pruners.NopPruner()
        # more TPE random-startup trials for the 8-param joint space before TPE models it
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=None,
                                             n_startup_trials=12)
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=None)
    # Storage: SQLite (single box) or JournalStorage on shared NFS (multi-box). SQLite
    # over NFS corrupts under cross-host POSIX locks; the journal's symlink lock is
    # NFS-safe, so 2+ instances can drive ONE distributed study.
    if args.journal:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import (JournalFileBackend,
                                             JournalFileSymlinkLock)
        os.makedirs(os.path.dirname(args.journal) or ".", exist_ok=True)
        storage = JournalStorage(
            JournalFileBackend(args.journal, lock_obj=JournalFileSymlinkLock(args.journal)))
        logger.info(f"storage: JournalStorage (NFS-safe symlink lock) @ {args.journal}")
    else:
        os.makedirs(os.path.dirname(args.db), exist_ok=True)
        storage = f"sqlite:///{args.db}"
        logger.info(f"storage: SQLite @ {args.db} (single-box only — NOT NFS-safe)")
    study = optuna.create_study(
        study_name=args.study, storage=storage,
        load_if_exists=True, direction="maximize",
        sampler=sampler, pruner=pruner)
    if args.search_awp and args.enqueue_anchor and not study.get_trials(deepcopy=False):
        study.enqueue_trial({  # champion recipe + AWP defaults = LB 0.78557 (E34);
            # 'epochs' omitted — it is fixed (=4), not a searched param for E35
            "lr": 2e-5, "eff_batch": 16, "warmup_ratio": 0.05,
            "label_smoothing": 0.10, "weight_decay": 0.01,
            "awp_gamma": 1e-3, "awp_lr": 1e-4, "awp_start_epoch": 1})
        logger.info("E35: enqueued LB-validated AWP config as anchor trial 0")
    study.optimize(lambda t: run_trial(t, args), n_trials=args.n_trials,
                   gc_after_trial=True)
    best = study.best_trial
    logger.success(f"worker {args.gpu_tag} done. best so far: "
                   f"F1={best.value:.4f} params={best.params}")


if __name__ == "__main__":
    main()
