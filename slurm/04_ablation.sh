#!/bin/bash
#SBATCH --job-name=xai_vd_abl
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

DS=${1:-rlvs}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# The TF 2.15 wheel carries no sm_90 cubins, so every kernel is JIT-compiled
# from PTX on an H100 - 30 min or more per process on a cold cache. Persist the
# compiled kernels in $HOME (visible from the compute nodes) so only the very
# first run pays for it.
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296

cd $HOME/xai-vd
mkdir -p logs results

python scripts/run_ablation.py \
    --features features/"$DS" \
    --seeds 5 \
    --out results/ablation_"$DS".json
