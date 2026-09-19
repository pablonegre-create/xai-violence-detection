#!/bin/bash
#SBATCH --job-name=xai_vd_probe
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Order probe on all five datasets. Nothing is retrained: it loads the cached
# descriptors and the trained head, permutes the frame axis and re-runs the
# head, so the whole sweep is a couple of minutes. It is the cheapest job in
# the repository and the one the paper's main claim rests on.
#
#   sbatch slurm/09_temporal_probe.sh

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
    # rlvs keeps the unsuffixed filename the first run used
    if [ "$DS" = "rlvs" ]; then
        OUT=results/temporal_probe.json
    else
        OUT=results/temporal_probe_${DS}.json
    fi

    echo "=== $DS ==="
    python scripts/run_temporal_probe.py \
        --features features/"$DS" \
        --head-weights ckpt/head_"$DS"_seed0.h5 \
        --out "$OUT"
done

echo
echo "verdicts:"
grep -h '"verdict"' results/temporal_probe*.json
