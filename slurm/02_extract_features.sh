#!/bin/bash
#SBATCH --job-name=xai_vd_feat
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Caches per-frame descriptors for one dataset.
#   for d in rlvs rwf2000 hockey movies violentflows; do
#       sbatch slurm/02_extract_features.sh $d
#   done
# Decoding is the bottleneck here, not the GPU.

set -euo pipefail

DS=${1:-rlvs}
DATA=${DATA_ROOT:-$HOME/data}/$DS
CKPT=${BACKBONE_CKPT:-$HOME/xai-vd/ckpt/mnv2_ft5.h5}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

cd $HOME/xai-vd
mkdir -p logs features

python scripts/extract_features.py \
    --dataset "$DS" \
    --root "$DATA" \
    --out features/"$DS" \
    --backbone-weights "$CKPT"
