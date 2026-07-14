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
    # epochs: FIXED for the AWP search (E38), still searched for E28 back-compat.
    # We RAM-snapshot the BEST-epoch checkpoint, which makes "more epochs" a free lunch
    # in the objective (more chances at a high best epoch) → a *searched* epochs drifts
    # to 5 by noise (observed in E28) without being genuinely better, and wastes compute.
    # Fixing epochs + best-epoch selection extracts each config's true peak; 4 contains
    # it (E28 curves peak at ep3; AWP regularizes so its peak may sit a touch later).
    if args.search_epochs:
        # E41-b: user wants epochs SEARCHED (not fixed at 4) to check whether some
        # qwen3 configs peak later than epoch 4 — the ep6 diagnostic only tested 2 of
        # 14 configs. Reintroduces the free-lunch bias noted below (more epochs = more
        # chances at a high best-epoch eval by noise); bounded to 3-6 (matches the
        # diagnostic's tested range) to cap the damage and total compute per trial.
        epochs = trial.suggest_int("epochs", args.epochs_min, args.epochs_max)
    elif args.search_drophead:
        # E50: PINNED at 5 (user 2026-07-15). DropHead is a regularizer that delays the
        # peak (6ep run peaked ep5; still climbing at ep4) — 4 would systematically clip
        # strong-regularization configs, 6 wastes ~25% compute (E47 anchor peaked ep3),
        # 8 is demonstrably schedule-depressed (E47 §schedule: −0.0057 for the same
        # config). Never searched (E28 best-epoch free-lunch drift).
        epochs = 5
    elif args.search_awp:
        epochs = 4
    else:
        epochs = trial.suggest_int("epochs", 2, 5)
    if args.search_drophead:
        # E50: DropHead-tree search JOINTLY with the recipe in NARROW bands around the
        # E38 t031 optimum (lr 3.47e-5 · warmup .144 · ls .137 · wd 1.7e-3 · γ 2.0e-3 ·
        # awp_lr 2.3e-4) — E38's own lesson: the recipe re-tunes under a new regularizer,
        # but the t031 neighborhood is the only LB-validated region, so stay near it.
        # DropHead dims per the E47/E49 two-bar analysis (2026-07-15): p continuous
        # (unimodal ~0.10-0.15, both sweep tails flat-to-neg), schedule {const,updown}
        # (updown cleared both bars + peaks early), layer_ramp (E49 standout: 0.7824 on
        # the depressed 8ep schedule), MSD (dh010+msd = wave-2 high). EXCLUDED as
        # decisively-negative: correlated, LayerDrop, DH+LayerDrop.
        lr = trial.suggest_float("lr", 1.5e-5, 6e-5, log=True)
        warmup = trial.suggest_float("warmup_ratio", 0.05, 0.20)
        ls_eps = trial.suggest_float("label_smoothing", 0.08, 0.18)
        wd = trial.suggest_float("weight_decay", 3e-4, 8e-3, log=True)
        eff_batch = 16
    elif args.narrow:
        # E39 (qwen3): transfer E38's granite findings as PRIORS to cut the space so a
        # small trial budget (1 trial/GPU, ~16-24 trials) converges. Fix the dims E38
        # settled — eff_batch=16 dominated, warmup~0.1, awp_start=1 — and search only the
        # 4 that matter, in narrow bands around the E38 optima.
        lr = trial.suggest_float("lr", 8e-6, 4e-5, log=True)
        warmup = 0.10
        # E41-b: user hypothesis — wd~0.01 (E38's granite optimum) may HURT qwen3;
        # reopen it and let 0 compete. Linear (not log) so 0.0 is a valid endpoint.
        wd = (trial.suggest_float("weight_decay", 0.0, 0.03) if args.search_wd else 0.01)
        # E41-b: reopen eff_batch too (E38's "16 dominates" finding was granite-only).
        eff_batch = (trial.suggest_categorical("eff_batch", [8, 16, 32])
                     if args.search_eff_batch else 16)
        ls_eps = trial.suggest_float("label_smoothing", 0.05, 0.18)
    else:
        lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
        eff_batch = trial.suggest_categorical("eff_batch", [8, 16, 32])
        warmup = trial.suggest_float("warmup_ratio", 0.0, 0.15)
        ls_eps = trial.suggest_float("label_smoothing", 0.02, 0.20)
        wd = trial.suggest_float("weight_decay", 1e-3, 0.1, log=True)
    # E19 fast shape: real batches up to the measured-safe bs16, accumulation only
    # for the eff-32 arm (bs32@512 unmeasured on 24GB -> bs16xga2, identical grads).
    # grad_accum pinned explicitly (finetune.py default is 4!); grad-ckpt auto-off.
    bs = min(eff_batch, 16)
    # E50 protocol: STANDARD split_indices 56k/14k (fold=-1), NOT the session fold —
    # every DropHead measurement (E47/E49, anchor 0.7782, DropHead 0.7826) is on the
    # standard split; the search must rank on the same yardstick.
    fold = -1 if args.search_drophead else args.fold
    cmd = [PY, "-u", "-m", "src.finetune",
           "--model", args.model, "--serialize", "richargs", "--loss", "ls",
           "--init_seed", "42", "--group_by_length",
           "--session_fold", str(fold), "--session_splits", "5",
           "--lr", f"{lr:.3e}", "--epochs", str(epochs),
           "--batch_size", str(bs), "--grad_accum", str(eff_batch // bs),
           "--warmup_ratio", f"{warmup:.4f}", "--label_smoothing", f"{ls_eps:.4f}",
           "--weight_decay", f"{wd:.4e}",
           "--tag", f"{args.exp}_t{trial.number:03d}",
           "--out_dir", args.out_dir,
           "--results_name", f"optuna_{args.gpu_tag}.csv"]  # per-worker CSV: no append race
    # E38: joint AWP-knob search on top of the recipe search. AWP was LB-validated
    # (0.78557, single-model SOTA) at the blind defaults gamma=1e-3/adv_lr=1e-4/
    # start_epoch=1.0 (E34) — that config is enqueued as the study anchor. start_epoch
    # is capped at {0,1} so AWP is active for the LATER epochs the pruner reports on
    # (it fires only from state.epoch>=start_epoch → epoch 2 hits AWP for BOTH values,
    # keeping MedianPruner meaningful; start_epoch=2 would leave 2 pre-AWP epochs that
    # look identical to a plain run and mis-prune).
    if args.search_awp or args.search_drophead:
        if args.search_drophead:  # E50: narrow AWP bands around the t031 optimum, start=1
            awp_gamma = trial.suggest_float("awp_gamma", 8e-4, 5e-3, log=True)
            awp_lr = trial.suggest_float("awp_lr", 1e-4, 4e-4, log=True)
            awp_start = 1
        elif args.narrow:  # E39: narrow AWP bands around E38 optima; start fixed at 1
            awp_gamma = trial.suggest_float("awp_gamma", 1e-3, 4e-3, log=True)
            awp_lr = trial.suggest_float("awp_lr", 1e-4, 2.8e-4, log=True)
            awp_start = 1
        else:
            awp_gamma = trial.suggest_float("awp_gamma", 5e-4, 5e-3, log=True)
            awp_lr = trial.suggest_float("awp_lr", 3e-5, 3e-4, log=True)
            awp_start = trial.suggest_categorical("awp_start_epoch", [0, 1])
        cmd += ["--awp_gamma", f"{awp_gamma:.4e}", "--awp_lr", f"{awp_lr:.4e}",
                "--awp_start_epoch", f"{float(awp_start):.1f}"]
    if args.search_drophead:
        # DropHead tree (E50). p=0 is NOT in the space — the no-DropHead baseline is
        # measured by the box-B seed replicates, not spent search trials.
        # ramp_mode splits the study per box (no shared storage): 'off' = constant-p,
        # p in [0.05, 0.20] (0.10 won, 0.15 cleared both bars, tails flat-neg).
        # 'on' = layer-ramp fixed ON; per-layer p = nominal*(i+1)/L means MEAN effective
        # p ~ nominal/2, so the nominal range is doubled to [0.08, 0.30] to cover the
        # same effective-regularization span (A1's range here would bias A2 weak).
        if args.ramp_mode == "on":
            dh_p = trial.suggest_float("drophead_p", 0.08, 0.30)
            cmd += ["--drophead_layer_ramp"]
        else:
            dh_p = trial.suggest_float("drophead_p", 0.05, 0.20)
        dh_sched = trial.suggest_categorical("drophead_schedule", ["const", "updown"])
        msd_k = trial.suggest_categorical("msd_k", [0, 4, 8])
        cmd += ["--drophead_p", f"{dh_p:.3f}", "--drophead_schedule", dh_sched]
        if msd_k > 0:
            msd_p = trial.suggest_float("msd_p", 0.10, 0.35)
            cmd += ["--msd_k", str(msd_k), "--msd_p", f"{msd_p:.3f}"]
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
                           f"ft_{args.model.replace('/', '__')}_{args.exp}_t{trial.number:03d}")
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
    ap.add_argument("--model", default=MODEL,
                    help="backbone to search (default granite; E39 = Qwen/Qwen3-Embedding-0.6B)")
    ap.add_argument("--narrow", action="store_true",
                    help="E39: prior-informed narrow space (fix eff_batch=16/warmup=0.1/"
                         "wd=0.01/awp_start=1, search only lr/ls/awp_gamma/awp_lr in narrow "
                         "bands around E38 optima) so a small trial budget converges fast")
    ap.add_argument("--search_awp", action="store_true",
                    help="E38: also search AWP knobs (gamma/adv_lr/start_epoch) "
                         "jointly with the recipe, on top of the champion anchor")
    ap.add_argument("--ramp_mode", default="off", choices=["off", "on"],
                    help="E50: which layer-ramp branch THIS box's study covers (A1=off "
                         "p[0.05,0.20]; A2=on p[0.08,0.30] — nominal doubled because "
                         "ramp mean-p ~ nominal/2)")
    ap.add_argument("--search_drophead", action="store_true",
                    help="E50: DropHead-tree search (p/schedule/layer_ramp/MSD) jointly "
                         "with narrow recipe+AWP bands around t031. Epochs PINNED at 5, "
                         "STANDARD 56k/14k split (fold forced -1), NopPruner. Anchors: "
                         "the E47/E49 named cells (const/ramp/msd at p=0.1)")
    ap.add_argument("--search_epochs", action="store_true",
                    help="E41-b: search epochs instead of fixing at 4 (see build_cmd "
                         "comment for the free-lunch bias this reintroduces); range set "
                         "by --epochs_min/--epochs_max")
    ap.add_argument("--epochs_min", type=int, default=3)
    ap.add_argument("--epochs_max", type=int, default=6)
    ap.add_argument("--search_wd", action="store_true",
                    help="E41-b: reopen weight_decay (narrow space fixes it at 0.01 from "
                         "E38's granite finding) — search [0.0, 0.03] linear, 0 included")
    ap.add_argument("--search_eff_batch", action="store_true",
                    help="E41-b: reopen eff_batch (narrow space fixes it at 16 from E38's "
                         "granite finding) — search {8, 16, 32}")
    ap.add_argument("--enqueue_anchor", action="store_true",
                    help="E38: enqueue the LB-validated config (champion recipe + AWP "
                         "gamma1e-3/lr1e-4/start1 = LB 0.78557) as the first trial so "
                         "every result reads as a Delta vs a known-LB point. Pass on "
                         "ONE worker only (avoids duplicate anchors)")
    args = ap.parse_args()
    log_cmd()

    assert cuda_alive(), \
        "cuInit failed — host driver is poisoned (see fleet memory); do not start trials"

    # Pruner: E28 used MedianPruner (kill trials below the median epoch-curve). For the
    # AWP search (E38) we DISABLE pruning (NopPruner): AWP's benefit appears only in the
    # LATER epochs (it activates from start_epoch), so early-epoch pruning would risk
    # killing a slow-start-but-blooms config — and with epochs fixed at 4 on 16x5090,
    # the compute saved by pruning is marginal vs that downside. Run every trial full.
    if args.search_drophead:
        # E50: no pruning (DropHead peaks ep4-5 — early-epoch pruning would kill exactly
        # the strong-regularization configs); group=True for the conditional msd_p dim.
        pruner = optuna.pruners.NopPruner()
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=None,
                                             n_startup_trials=12)
    elif args.search_awp:
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
    if args.search_drophead and args.enqueue_anchor and not study.get_trials(deepcopy=False):
        # E50 anchors = t031 recipe + the named E47/E49 cells for THIS ramp branch, so
        # the study starts from known-good points and every result reads as a Δ vs them.
        t031 = {"lr": 3.466e-5, "warmup_ratio": 0.1441, "label_smoothing": 0.1365,
                "weight_decay": 1.7158e-3, "awp_gamma": 2.0116e-3, "awp_lr": 2.309e-4}
        if args.ramp_mode == "on":   # A2
            study.enqueue_trial({**t031, "drophead_p": 0.10, "drophead_schedule": "const",
                                 "msd_k": 0})                              # E49 S8 cell exact
            study.enqueue_trial({**t031, "drophead_p": 0.20, "drophead_schedule": "const",
                                 "msd_k": 0})                              # strength-matched twin of A1 winner
            study.enqueue_trial({**t031, "drophead_p": 0.20, "drophead_schedule": "const",
                                 "msd_k": 4, "msd_p": 0.30})               # ramp+MSD (untested stack)
        else:                        # A1
            study.enqueue_trial({**t031, "drophead_p": 0.10, "drophead_schedule": "const",
                                 "msd_k": 0})                              # E47 winner cell
            study.enqueue_trial({**t031, "drophead_p": 0.10, "drophead_schedule": "const",
                                 "msd_k": 4, "msd_p": 0.30})               # wave-2 dh+msd high
            study.enqueue_trial({**t031, "drophead_p": 0.15, "drophead_schedule": "const",
                                 "msd_k": 0})                              # wave-2 best-p
        logger.info(f"enqueued 3 E50 anchors (ramp_mode={args.ramp_mode})")
    if args.search_awp and args.enqueue_anchor and not study.get_trials(deepcopy=False):
        if args.narrow:  # E39: only the 4 searched keys exist in the narrow space
            anchor = {"lr": 2e-5, "label_smoothing": 0.10,
                      "awp_gamma": 1e-3, "awp_lr": 1e-4}
        else:  # champion recipe + AWP defaults = LB 0.78557 (E34); 'epochs' fixed (=4)
            anchor = {"lr": 2e-5, "eff_batch": 16, "warmup_ratio": 0.05,
                      "label_smoothing": 0.10, "weight_decay": 0.01,
                      "awp_gamma": 1e-3, "awp_lr": 1e-4, "awp_start_epoch": 1}
        study.enqueue_trial(anchor)
        logger.info("enqueued anchor trial 0 (known-good starting point)")
    study.optimize(lambda t: run_trial(t, args), n_trials=args.n_trials,
                   gc_after_trial=True)
    best = study.best_trial
    logger.success(f"worker {args.gpu_tag} done. best so far: "
                   f"F1={best.value:.4f} params={best.params}")


if __name__ == "__main__":
    main()
