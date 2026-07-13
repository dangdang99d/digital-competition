#!/bin/bash
#SBATCH --job-name=e26-teachers
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# E26 path-1B: harvest the two missing member logit caches (e8a champion ~20min,
# e8b qwen3_ls ~40min) and assemble all three teacher npzs. The three trio caches
# (is3 / aum06 / e25c) must already exist in analysis/cache/ (synced from ocean;
# they are NOT rebuildable here without those checkpoints).

set -euo pipefail
cd /NAS2/ksjang0515/dacon

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
mkdir -p analysis/cache sbatch/logs

for f in e26_train_logits_e12soup_granite_ls_is3 \
         e26_train_logits_e22_aum06_champion \
         e26_train_logits_e25c_richmeta_ls_bf16; do
  [ -f "analysis/cache/${f}.npz" ] || { echo "MISSING analysis/cache/${f}.npz — sync it from ocean first"; exit 1; }
done

# T2 = LB-certified trio + champion + qwen3_ls (harvests e8a/e8b, reuses trio caches)
PYTHONPATH=. "${PY}" -u experiments/ensemble/distill_teacher.py \
  --members e12soup_granite_ls_is3,e22_aum06_champion,e25c_richmeta_ls_bf16,e8a_ls_richargs_full,e8b_ls_qwen3_richargs_full \
  --out analysis/cache/e26_teacher_T2_5mem.npz

# born-again control teacher = champion alone (cache now exists -> instant, CPU)
PYTHONPATH=. "${PY}" -u experiments/ensemble/distill_teacher.py \
  --members e8a_ls_richargs_full \
  --out analysis/cache/e26_teacher_BA_champ.npz

# T1 trio teacher (already built on ocean; rebuild from caches if the npz didn't sync)
[ -f analysis/cache/e26_teacher_T1_trio.npz ] || PYTHONPATH=. "${PY}" -u experiments/ensemble/distill_teacher.py \
  --members e12soup_granite_ls_is3,e22_aum06_champion,e25c_richmeta_ls_bf16 \
  --out analysis/cache/e26_teacher_T1_trio.npz

echo "teachers ready:"; ls -la analysis/cache/e26_teacher_*.npz
