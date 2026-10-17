#!/bin/bash
#SBATCH --job-name=xai_vd_kfold
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 10-fold cross-validation on all five datasets, cut over the cached
# descriptors so nothing is re-extracted. Folds keep duplicate groups together.
# Also re-runs the per-dataset training so the saved predictions carry
# probabilities, which is what an ROC curve needs.
#
#   sbatch slurm/12_kfold.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296

cd $HOME/xai-vd
mkdir -p logs results

for DS in rlvs rwf2000 hockey movies violentflows; do
    echo
    echo "=== 10-fold: $DS ==="
    python scripts/run_kfold.py \
        --features features/"$DS" --dataset "$DS" --k 10 \
        --out results/kfold_"$DS".json
done

for DS in rlvs rwf2000 hockey movies violentflows; do
    echo
    echo "=== probabilities for the ROC curve: $DS ==="
    python scripts/train.py \
        --features features/"$DS" --seeds 5 \
        --save-preds results/preds_"$DS".json \
        --out results/main_"$DS".json
done
