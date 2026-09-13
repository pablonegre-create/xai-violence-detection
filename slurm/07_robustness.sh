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
