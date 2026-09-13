#!/bin/bash
#SBATCH --job-name=xai_vd_ft
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fine-tunes MobileNetV2 on frames, one job per depth so the fine-tuning
# ablation can run in parallel:
#   for d in 0 5 all; do sbatch slurm/01_finetune_backbone.sh $d; done

set -euo pipefail

DEPTH=${1:-5}
DATA=${DATA_ROOT:-$HOME/data/RLVS}
OUT=$HOME/xai-vd/ckpt/mnv2_ft${DEPTH}.h5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "node $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}, depth=$DEPTH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $HOME/xai-vd
mkdir -p logs ckpt

python scripts/finetune_backbone.py \
    --root "$DATA" \
    --finetune "$DEPTH" \
    --out "$OUT" \
    --epochs 15 --steps 200 --batch 32
