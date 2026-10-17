#!/bin/bash
#SBATCH --job-name=xai_vd_abl2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The ablation again with twenty seeds instead of five, and with an equivalence
# test. A large p-value never showed that mean pooling matches the Bi-LSTM; it
# only showed the five seeds could not tell them apart. TOST against a stated
# margin can make the positive claim, or fail to, and either way the confidence
# interval on the paired difference says what the data still allow.
#
#   sbatch slurm/15_ablation_seeds.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296

cd $HOME/xai-vd
mkdir -p logs results

python scripts/run_ablation.py \
    --features features/rlvs \
    --seeds 20 \
    --equivalence-margin 0.01 0.02 \
    --tmp results/_ablation20_tmp \
    --out results/ablation_rlvs.json
