#!/bin/bash
#SBATCH --job-name=bgem3-disc
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# Within-group discrimination experiments (explore-group confusions = 66% of all
# errors; baseline hist0 = 0.7568 cal). Four independent boundary-sharpening runs:
#   task 0  rdrop:    R-Drop consistency (2x fwd + sym-KL), fresh 3ep. ~2x step time.
#   task 1  supcon:   SupCon aux on pooled emb, XBM queue, same-group negatives x2,
#                     fresh 3ep. Projection head is train-only.
#   task 2  hardex:   hard-example continuation from hist0: mine boundary samples
#                     (same-group top-2, margin<2) with the checkpoint, upweight x3,
#                     +1ep @ lr 1e-5.
#   task 3  pair:     grep_search-vs-read_file binary specialist from hist0 encoder
#                     (biggest single error pair). Composed via margin-gated deferral
#                     later (local analysis) — this run just trains + reports 2-class F1.
#   Results: output/pat/ft_results_disc_<task>.csv

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

HIST0=output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500
TAGS=(rdrop supcon hardex pair_grepread)
EPOCHS=(3 3 1 2)
LRS=(2e-5 2e-5 1e-5 1e-5)
INITS=("" "" "${HIST0}" "${HIST0}")
EXTRA=("--rdrop 1.0"
       "--supcon 0.3 --supcon_tau 0.1 --supcon_group_w 2.0"
       "--hard_boundary 3.0 --boundary_margin 2.0"
       "--pair grep_search,read_file")
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
INIT="${INITS[$SLURM_ARRAY_TASK_ID]}"

if [[ -n "${INIT}" && ! -f "${INIT}/model.safetensors" ]]; then
  echo "ERROR: init checkpoint missing on pat: ${INIT}"
  exit 1
fi

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): tag=${TAG}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

INIT_ARG=()
if [[ -n "${INIT}" ]]; then INIT_ARG=(--init_from "${INIT}"); fi

# shellcheck disable=SC2086  # EXTRA is intentionally word-split
"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize v1 \
  --tag "${TAG}" \
  "${INIT_ARG[@]}" \
  --max_len 1024 --epochs "${EPOCHS[$SLURM_ARRAY_TASK_ID]}" \
  --lr "${LRS[$SLURM_ARRAY_TASK_ID]}" \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_disc_${SLURM_ARRAY_TASK_ID}.csv" \
  ${EXTRA[$SLURM_ARRAY_TASK_ID]}
