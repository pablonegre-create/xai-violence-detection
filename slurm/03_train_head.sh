#!/bin/bash
#SBATCH --job-name=xai_vd_train
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The head trains on cached features, so this is minutes, not hours.
#   sbatch slurm/03_train_head.sh rlvs

set -euo pipefail

DS=${1:-rlvs}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

cd $HOME/xai-vd
mkdir -p logs results

python scripts/train.py \
    --features features/"$DS" \
    --seeds 5 \
    --out results/main_"$DS".json \
    --save-preds results/preds_"$DS".json
