#!/usr/bin/bash
#SBATCH -J dacon-tren
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o /data/kimin10866/repos/데이콘/results/slurm-%A.out

# 공정한 번역 실험: 전체 train을 영어로 번역(4-GPU 병렬) -> granite 재학습(2-fold) -> OOF 비교(SOTA 0.7725).
set -uo pipefail
source /data/kimin10866/anaconda3/etc/profile.d/conda.sh
conda activate dacon
PROJ=/data/kimin10866/repos/데이콘; cd "$PROJ"
export HF_HOME=/data/kimin10866/hf_home TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
BASE="$PROJ/model/base/granite-embedding-311m-multilingual-r2"; NB="$PROJ/train_and_analyze_4gpu.ipynb"
RES="$PROJ/results/translate_retrain_${SLURM_JOB_ID}"; mkdir -p "$RES/logs" "$RES/notebooks"
echo JOB START; date; hostname; nvidia-smi -L
pip install -q sentencepiece sacremoses 2>&1 | tail -1
echo "=== 노트북 재생성 (compute 노드에서) ==="
python "$PROJ/build_nb.py" "$NB"
rm -rf "$PROJ/results/tmp_translate"

echo "=== 단계1: 번역 (GPU 4대 병렬) ==="
for sh in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$sh SHARD=$sh NSHARD=4 python "$PROJ/translate_shard.py" > "$RES/logs/shard_${sh}.out" 2>&1 &
done
wait
echo "번역 완료:"; grep -h "DONE" "$RES"/logs/shard_*.out
date

echo "=== 단계2: 병합 + data_en 생성 ==="
python "$PROJ/build_translated_data.py" > "$RES/logs/build.out" 2>&1
tail -3 "$RES/logs/build.out"

echo "=== 단계3: granite 재학습 (data_en, 2-fold, GPU 0/1) ==="
run_fold () {
  local k=$1 gpu=$2
  local pf="$RES/logs/params_fold${k}.yaml"
  { printf 'DEV_MODE: false\nFOLD: %s\nRUN_FINALIZE: false\n' "$k"
    printf 'RESULTS_DIR: "%s"\n' "$RES"; printf 'BASE_MODEL_PATH: "%s"\n' "$BASE"
    printf 'DATA_DIR_P: "%s"\nOPENFILES: names\n' "$PROJ/data_en"; } > "$pf"
  CUDA_VISIBLE_DEVICES=$gpu papermill "$NB" "$RES/notebooks/train_fold${k}.ipynb" -f "$pf" --log-output > "$RES/logs/fold${k}.out" 2>&1
}
run_fold 0 0 & p0=$!
run_fold 1 1 & p1=$!
wait $p0 || true; wait $p1 || true
date

echo "=== 단계4: finalize (OOF+bias) ==="
pf="$RES/logs/params_finalize.yaml"
{ printf 'DEV_MODE: false\nFOLD: null\nRUN_FINALIZE: true\n'
  printf 'RESULTS_DIR: "%s"\n' "$RES"; printf 'BASE_MODEL_PATH: "%s"\n' "$BASE"
  printf 'DATA_DIR_P: "%s"\nOPENFILES: names\nFOLDS_TO_FINALIZE: [0, 1]\n' "$PROJ/data_en"; } > "$pf"
CUDA_VISIBLE_DEVICES=0 papermill "$NB" "$RES/notebooks/finalize.ipynb" -f "$pf" --log-output > "$RES/logs/finalize.out" 2>&1 || true
rm -rf "$RES/trainer_tmp"

echo "=== 결과 (영어 재학습, 2-fold, tuned) ==="
grep -h "val macro-F1" "$RES"/logs/fold*.out 2>/dev/null
grep -h "OOF Macro-F1" "$RES"/logs/finalize.out 2>/dev/null
echo "[비교] SOTA(한국어 원본)=0.7725"
echo JOB END; date
