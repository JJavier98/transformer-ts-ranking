#!/bin/bash
# M4 benchmark — lag_llama_pretrained only.
#
# Dedicated job to avoid blocking the main m4-full array.
# Autoregressive decoding makes this model ~pred_len× slower than others.
#
# Task layout (18 tasks = 6 frequencies × 3 seeds):
#   0-2   Yearly     (pred_len=6)   seeds 42, 123, 2026
#   3-5   Quarterly  (pred_len=8)   seeds 42, 123, 2026
#   6-8   Monthly    (pred_len=18)  seeds 42, 123, 2026
#   9-11  Weekly     (pred_len=13)  seeds 42, 123, 2026
#   12-14 Daily      (pred_len=14)  seeds 42, 123, 2026
#   15-17 Hourly     (pred_len=48)  seeds 42, 123, 2026
#
# Results layout:
#   results/raw/m4/lag_llama_pretrained_<FREQ>_s<SEED>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lag_llama_pretrained_m4.sh

#SBATCH --job-name lag-llama-m4
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lag_llama_m4_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lag_llama_m4_%A_%a.err

FREQUENCIES=(
    Yearly    Yearly    Yearly
    Quarterly Quarterly Quarterly
    Monthly   Monthly   Monthly
    Weekly    Weekly    Weekly
    Daily     Daily     Daily
    Hourly    Hourly    Hourly
)
SEEDS=(
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
)

T=$SLURM_ARRAY_TASK_ID
FREQ="${FREQUENCIES[$T]}"
SEED="${SEEDS[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lag_llama_pretrained M4: $FREQ seed=$SEED (zero-shot)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"
echo "WARNING: autoregressive decoding + univariate loop over series — expect slow runtime"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    training.batch_size=16 \
    experiment=m4 \
    "experiment.datasets=[$FREQ]" \
    "models=[lag_llama_pretrained]" \
    wandb=disabled \
    "+results_dir=results/raw/m4/lag_llama_pretrained_${FREQ}_s${SEED}" \
    "hydra.run.dir=outputs/m4/lag_llama_pretrained_${FREQ}_s${SEED}"

echo "[$SLURM_JOB_ID/$T] Finished: lag_llama_pretrained M4 $FREQ seed=$SEED"
