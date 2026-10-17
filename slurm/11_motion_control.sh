#!/bin/bash
#SBATCH --job-name=xai_vd_motion
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The matched motion control. Caches a second set of descriptors from
# frame-to-frame differences, then trains the same head on appearance, on
# motion, and on both, over the same split and the same seeds.
#
#   sbatch slurm/11_motion_control.sh rwf2000
#   sbatch slurm/11_motion_control.sh rlvs
#
# RWF-2000 is the dataset the argument is about; RLVS is the control where
# appearance already separates the classes, so motion should add little.

set -euo pipefail

DS=${1:-rwf2000}
DATA=${DATA_ROOT:-$HOME/data}/$DS

if [ ! -d "$DATA" ]; then
    echo "dataset directory not found: $DATA" >&2
    exit 1
fi

# each dataset uses the backbone its appearance features were built with, so
# the comparison changes the input and nothing else
if [ "$DS" = "rwf2000" ]; then
    BACKBONE=$HOME/xai-vd/ckpt/mnv2_ft5_rwf2000.h5
else
    BACKBONE=$HOME/xai-vd/ckpt/mnv2_ft5.h5
fi
if [ ! -s "$BACKBONE" ]; then
    echo "backbone checkpoint missing: $BACKBONE" >&2
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296

cd $HOME/xai-vd
mkdir -p logs results features

if [ ! -s "features/${DS}_diff/features.npy" ]; then
    echo "=== caching frame-difference descriptors for $DS ==="
    python scripts/extract_features.py \
        --dataset "$DS" --root "$DATA" \
        --input-mode diff \
        --backbone-weights "$BACKBONE" \
        --out features/"${DS}_diff"
else
    echo "frame-difference cache already present, reusing it"
fi

echo
echo "=== motion control on $DS ==="
python scripts/run_motion_control.py \
    --dataset "$DS" \
    --features-rgb features/"$DS" \
    --features-diff features/"${DS}_diff" \
    --seeds 5 --n-perm 10 \
    --out results/motion_control_"$DS".json
