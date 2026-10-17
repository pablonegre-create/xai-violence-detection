#!/bin/bash
#SBATCH --job-name=xai_vd_probe2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Order probe reporting what happens to the decision function, not only to
# accuracy: 20 independent permutations per dataset, prediction flip rates and
# per-clip probability shifts. Nothing is retrained.
#
#   sbatch slurm/10_probe_extended.sh

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
    if [ "$DS" = "rlvs" ]; then OUT=results/temporal_probe.json
    else OUT=results/temporal_probe_${DS}.json; fi

    echo "=== $DS ==="
    python scripts/run_temporal_probe.py \
        --features features/"$DS" \
        --head-weights ckpt/head_"$DS"_seed0.h5 \
        --n-perm 20 \
        --out "$OUT"
done

echo
echo "verdicts:"
grep -h '"verdict"' results/temporal_probe*.json
