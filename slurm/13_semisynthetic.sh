#!/bin/bash
#SBATCH --job-name=xai_vd_semisyn
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Attribution scored against a known event planted in real footage: a real
# non-violent RLVS clip with a window of frames spliced in from a donor clip,
# violent for positives and non-violent for negatives, so the cut itself tells
# the head nothing and only the window content does.
#
#   sbatch slurm/13_semisynthetic.sh

set -euo pipefail

DATA=${DATA_ROOT:-$HOME/data}/rlvs
BACKBONE=$HOME/xai-vd/ckpt/mnv2_ft5.h5

for p in "$DATA" "$BACKBONE"; do
    if [ ! -e "$p" ]; then echo "missing: $p" >&2; exit 1; fi
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_CACHE_MAXSIZE=4294967296

cd $HOME/xai-vd
mkdir -p logs results features

if [ ! -s "features/semisynth_rlvs/features.npy" ]; then
    echo "=== splicing clips and caching descriptors ==="
    python scripts/run_semisynthetic_xai.py build \
        --root "$DATA" --backbone-weights "$BACKBONE" \
        --n 600 --event-len 5 \
        --out features/semisynth_rlvs
else
    echo "spliced cache already present, reusing it"
fi

echo
echo "=== scoring the attribution methods against the planted window ==="
python scripts/run_semisynthetic_xai.py score \
    --data features/semisynth_rlvs \
    --explain 60 --block 5 \
    --out results/semisynthetic_xai.json
