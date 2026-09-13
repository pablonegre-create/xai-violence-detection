#!/bin/bash
#SBATCH --job-name=xai_vd_cx
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Params/FLOPs/latency on the H100, to sit next to the CPU-only numbers
# measured on a laptop.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xai_vd
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

cd $HOME/xai-vd
mkdir -p logs results

python scripts/measure_complexity.py --runs 100 --out results/complexity_h100.json
python scripts/measure_complexity.py --cpu --runs 50 --out results/complexity_cn001_cpu.json
