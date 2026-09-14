#!/bin/bash
#SBATCH --job-name=xai_vd_rob
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Every corruption forces a full re-decode plus a full backbone pass, so this
# is by far the longest job. 10 corruptions x 3 severities x 300 clips.

set -euo pipefail

DS=${1:-rlvs}
DATA=${DATA_ROOT:-$HOME/data}/$DS

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

python scripts/run_robustness.py \
    --root "$DATA" \
    --backbone-weights ckpt/mnv2_ft5.h5 \
    --head-weights ckpt/head_"$DS"_seed0.h5 \
    --severities 1 3 5 \
    --subset 300 \
    --adversarial \
    --out results/robustness_"$DS".json
