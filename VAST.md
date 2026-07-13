# Vast.ai training workflow

Rented cloud GPUs for training when the ocean fleet is saturated. Set up 2026-07-12.

## Setup (done once, already in place)

| What | Where |
|---|---|
| CLI | `vastai==1.3.0` in the `dacon` conda env (pinned in environment.yaml) |
| API key | `VAST_API_KEY` in project `.env` (gitignored) — `set -a; source .env; set +a` before `vastai` calls |
| SSH key | `.vast/vast_ed25519` (passwordless ed25519, gitignored, NFS-shared) — registered account-wide on vast (key id 1082102) |
| GitHub access | repo is **public** → instances clone anonymously over https, no deploy key. ⚠️ If flipped private (see data-redistribution note below), add `.vast/vast_ed25519.pub` as a read-only deploy key and clone via ssh |
| Data | raw `data/train.jsonl` (98.51MB) is committed — **1.5% under GitHub's 100MiB hard cap**; if it grows, pushes bounce. No git-LFS (1GB/mo bandwidth quota breaks repeated clones). ⚠️ **Repo is public → committed competition data is publicly redistributed, likely against DACON rules.** Resolve: make repo private (+ deploy key) or drop the data commit |
| Status script | `./vast_status.sh` — instance table + per-instance STATUS/GPU/log-tail (handles tqdm `\r` bars) |
| Template | `dacon-3090-vast` (the only one), hash `05cadc96453c5b8b1dd515ca280ffa62`: **`vastai/pytorch:2.7.1-cuda-12.8.1-py311-24.04-2026-06-15`** — Vast's maintained image family (hosts pre-cache it) **pinned to the exact ocean stack: torch 2.7.1 / CUDA 12.8.1 / python 3.11**. 🚫 NEVER use `@vastai-automatic-tag` or `:latest` — floating tags = indeterminate per-host versions, burned twice on 2026-07-12. Config: ssh-direct + 40GB + 3090 filter; bootstrap = `-e PROVISIONING_SCRIPT=<raw github url of vast_onstart.sh>` run after the image's own `entrypoint.sh` (do NOT override onstart — it does ssh/portal init). `vast_onstart.sh` lives in-repo (branch `research/token-selection`) → **edit the script in-repo, no template recreation**; it clones repo+data, pip-installs (skips the torch line — image already ships torch), touches `/workspace/READY`. tqdm bars stay ON (2026-07-12: vast_status handles \r; bar carries ETA) |

## GPU choice

**RTX 3090** (~$0.12–0.15/hr): identical to the ocean fleet (24GB Ampere) so batch sizes, bf16
recipe, and per-epoch timings port unchanged; ~3× cheaper than a 4090 for ~2× less speed → wins
per-dollar. Rules: **Ampere or newer only** (pre-Ampere T4/V100/2080Ti have no bf16 → silently
breaks the LS+bf16 champion recipe). **Avoid CN hosts** (HuggingFace unreliable there). Filter:

```bash
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 reliability>0.99 inet_down>500 disk_space>60' -o 'dph'
```

## Lifecycle per run

**⚠️ STEP 0 — MANDATORY RENTAL PROPOSAL (no exceptions).** Before creating ANY instance,
present the user a proposal table and get explicit approval. One row per instance:

| # | label | GPU / disk | experiment & run (exact tag) | est. runtime | est. cost |
|---|-------|-----------|------------------------------|--------------|-----------|

plus a TOTAL cost line. Only after the user approves the table may `vastai create instance`
be called. Vast is for real training runs only — never tests, smoke runs, or runs that might
already exist elsewhere (confirm completion state with the user first).

```bash
# 1. rent from the template (billing starts at boot, per second); pick offer manually
#    to keep control over price/geo (avoid CN hosts)
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 reliability>0.99 inet_down>500 disk_space>40' -o 'dph'
# --disk 40 is REQUIRED: the CLI ignores the template's disk setting (console-only) and
#   defaults to 10GB without it — verified 2026-07-12 (instance 40342499 came up 10GB)
vastai create instance <OFFER_ID> --template_hash 05cadc96453c5b8b1dd515ca280ffa62 --disk 40

# 2. label it IMMEDIATELY — experiment number + run tag, so `vast_status.sh` and the
#    web console say what each instance is running (vital once >1 instance is up)
vastai label instance <INSTANCE_ID> "E26_A_e26s_t1_a70_T2"   # <EXP>_<letter>_<run tag>

# 3. connect info
vastai show instances        # or ./vast_status.sh
vastai ssh-url <INSTANCE_ID>

# 4. bootstrap runs AUTOMATICALLY via the template onstart (clone + pip install,
#    ~2-3 min after boot). Wait for its READY sentinel before launching:
until ssh -i .vast/vast_ed25519 -p <PORT> root@<HOST> -o StrictHostKeyChecking=accept-new \
  'test -f /workspace/READY'; do sleep 20; done
# (debug a failed bootstrap: cat /workspace/onstart.log on the instance)

# 5. launch — sentinel convention (vast_status.sh and the watcher depend on it)
ssh ... 'cd /workspace/repo && TQDM_DISABLE=1 nohup sh -c \
  "python src/finetune.py ... > /workspace/train.log 2>&1; echo \$? > /workspace/DONE" \
  > /dev/null 2>&1 &'

# 6. watch — poll loop as a background task; exits when DONE appears
until ssh -i .vast/vast_ed25519 -p <PORT> root@<HOST> 'test -f /workspace/DONE'; do sleep 300; done

# 7. results DOWN FIRST, destroy SECOND (destroy erases the disk)
rsync -az -e "ssh -i .vast/vast_ed25519 -p <PORT>" root@<HOST>:/workspace/repo/runs/<run>/ runs/<run>/
vastai destroy instance <INSTANCE_ID>
```

## Conventions

- **Label every instance at rent time**: `<EXP>_<letter>_<run tag>` (e.g.
  `E26_A_e26s_t1_a70_T2`) — experiment number first, then a short A/B/C/… letter for easy
  reference in conversation, then the run tag. The label is the only way to tell instances
  apart in `vast_status.sh` / the console; an unlabeled instance is a mystery bill.
- **Instances are cattle**: nothing persists after destroy; anything you care about must be
  rsynced back to NFS first. A *stopped* (not destroyed) instance still bills storage and its
  GPU can be rented out from under you — always destroy.
- **Sentinel**: every run writes `/workspace/DONE` containing the exit code (0 = success).
  Presence = finished; contents distinguish success from crash. `vast_status.sh` reads it.
- **Logs**: `/workspace/*.log`, `TQDM_DISABLE=1` at launch (tqdm ≥4.66 honors it) so the log
  carries the Trainer's every-50-step `{loss, epoch}` lines instead of `\r` bar spam.
- **On-demand, not interruptible** for multi-hour runs; interruptible (~40% cheaper) only for
  short experiments with per-epoch checkpointing.

## Getting trained weights TO an instance

| Case | Channel |
|---|---|
| One-off checkpoint (~GB) | `rsync -az --partial` straight up from NFS |
| Weights reused across instances (e.g. distillation teachers) | private HF Hub repo — upload once, instances pull at datacenter speed |
| Instance → instance | `vastai copy` (never touches our uplink) |
| Distillation soft targets | **don't ship teacher weights at all** — `src/dump_logits.py` → fp16 logits over 70k rows ≈ a few MB, scp or commit |
