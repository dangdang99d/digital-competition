#!/bin/bash
#SBATCH --job-name=e26-students
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# E26 path-1B student run set (user-approved plan, experiments/ensemble/results.md):
#   0  T1 trio    vanilla KD  alpha=0.7 T=2   (workhorse)
#   1  T1 trio    soft-labels-only alpha=1 T=1 (teacher-as-better-labels)
#   2  T2 5-mem   vanilla KD  alpha=0.7 T=2   (cross-backbone: qwen3 -> granite)
#   3  champion   vanilla KD  alpha=0.7 T=2   (born-again CONTROL)
# Student = FROM-SCRATCH granite champion recipe (E24: never warm-start); no --loss ls
# (not combinable with --distill_from; KD soft targets act as learned label smoothing).
# Tasks 0-1 need only e26_teacher_T1_trio.npz; 2-3 need the e26-teachers job first.

set -euo pipefail
cd /NAS2/ksjang0515/dacon

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python

TEACHERS=(e26_teacher_T1_trio.npz e26_teacher_T1_trio.npz e26_teacher_T2_5mem.npz e26_teacher_BA_champ.npz)
ALPHAS=(0.7 1.0 0.7 0.7)
TEMPS=(2.0 1.0 2.0 2.0)
TAGS=(e26s_t1_a70_T2 e26s_t1_soft e26s_t2_a70_T2 e26s_ba_a70_T2)

i=${SLURM_ARRAY_TASK_ID}
TEACHER=analysis/cache/${TEACHERS[$i]}
[ -f "${TEACHER}" ] || { echo "MISSING ${TEACHER} — run sbatch/e26_teachers.sbatch first"; exit 1; }

"${PY}" -u -m src.finetune \
  --distill_from "${TEACHER}" --distill_alpha "${ALPHAS[$i]}" --distill_T "${TEMPS[$i]}" \
  --tag "${TAGS[$i]}" \
  --model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs --full_data \
  --epochs 3 --lr 2e-5 --batch_size 16 --grad_accum 1 --group_by_length --max_len 512 --seed 42 \
  --out_dir ./output/pat --results_name ft_results_e26.csv
