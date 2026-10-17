#!/bin/bash
#SBATCH --job-name=xai_vd_gcam
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Redraw the Grad-CAM panel from the MobileNetV2 the paper reports. One clip
# and one backbone pass, so it asks for no GPU and does not queue behind the
# training jobs.
#
#   sbatch slurm/16_gradcam_figure.sh

set -euo pipefail

DATA=${DATA_ROOT:-$HOME/data}/hockey
CLS=$HOME/xai-vd/ckpt/mnv2_ft5_cls.h5

for p in "$DATA" "$CLS"; do
    if [ ! -e "$p" ]; then echo "missing: $p" >&2; exit 1; fi
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

cd $HOME/xai-vd
mkdir -p logs figures

python scripts/make_gradcam_figure.py \
    --root "$DATA" \
    --backbone-weights "$CLS" \
    --out figures/gradcam_mobilenetv2
