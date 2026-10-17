#!/bin/bash
#SBATCH --job-name=xai_vd_indus
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The Industrial Surveillance Dataset of Ullah et al. (IEEE TII 2022), 300
# clips of real industrial CCTV, released with their VD_ConvLSTM_GRU
# repository. A sixth benchmark, and the one closest in character to RWF-2000,
# which is where our result bites.
#
# Fetch it first from the login node; compute nodes have no outbound network:
#   git clone --depth 1 https://github.com/FathUMinUllah3797/VD_ConvLSTM_GRU.git \
#       ~/data/_indus_src
#   mkdir -p ~/data/industrial
#   cp -r "~/data/_indus_src/Industrial Surveillance Dataset/Violent" \
#         ~/data/industrial/
#   cp -r "~/data/_indus_src/Industrial Surveillance Dataset/Non-Violent" \
#         ~/data/industrial/
#
#   sbatch slurm/14_industrial.sh

set -euo pipefail

DS=industrial
DATA=${DATA_ROOT:-$HOME/data}/$DS
BACKBONE=$HOME/xai-vd/ckpt/mnv2_ft5.h5

if [ ! -d "$DATA" ]; then
    echo "dataset directory not found: $DATA" >&2
    echo "see the header of this file for how to fetch it" >&2
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

echo "=== integrity check ==="
python scripts/check_dataset_integrity.py "$DATA" || true

if [ ! -s "features/${DS}/features.npy" ]; then
    echo "=== caching appearance descriptors ==="
    python scripts/extract_features.py \
        --dataset "$DS" --root "$DATA" \
        --backbone-weights "$BACKBONE" --out features/"$DS"
fi

if [ ! -s "features/${DS}_diff/features.npy" ]; then
    echo "=== caching frame-difference descriptors ==="
    python scripts/extract_features.py \
        --dataset "$DS" --root "$DATA" --input-mode diff \
        --backbone-weights "$BACKBONE" --out features/"${DS}_diff"
fi

echo
echo "=== detection performance ==="
python scripts/train.py --features features/"$DS" --seeds 5 \
    --ckpt-prefix ckpt/head_"$DS" \
    --save-preds results/preds_"$DS".json \
    --out results/main_"$DS".json

echo
echo "=== 10-fold ==="
python scripts/run_kfold.py --features features/"$DS" --dataset "$DS" --k 10 \
    --out results/kfold_"$DS".json

echo
echo "=== order probe ==="
python scripts/run_temporal_probe.py \
    --features features/"$DS" --head-weights ckpt/head_"${DS}"_seed0.h5 \
    --n-perm 20 --out results/temporal_probe_"$DS".json

echo
echo "=== motion control ==="
python scripts/run_motion_control.py --dataset "$DS" \
    --features-rgb features/"$DS" --features-diff features/"${DS}_diff" \
    --seeds 5 --n-perm 10 \
    --out results/motion_control_"$DS".json
