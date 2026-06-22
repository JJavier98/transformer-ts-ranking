#!/bin/bash
# Long-term benchmark — lag_llama_pretrained only.
#
# This model is run in a dedicated job because its autoregressive decoding
# (~pred_len sequential forward passes) makes it ~96-720× slower than any
# other model.  Isolating it prevents it from blocking the main experiment
# array jobs.
#
# Each task covers ONE dataset × ONE seed × all 4 horizons (3 seeds per
# dataset ensures the 4-day wall-time is respected even for slow datasets).
#
# Task layout (27 tasks total):
#   illness       : tasks 0-2   (seeds 42, 123, 2026) — horizons [24,36,48,60]
#   exchange_rate : tasks 3-5   (seeds 42, 123, 2026) — horizons [96,192,336,720]
#   ETTh1         : tasks 6-8   (seeds 42, 123, 2026)
#   ETTh2         : tasks 9-11  (seeds 42, 123, 2026)
#   weather       : tasks 12-14 (seeds 42, 123, 2026)
#   ETTm1         : tasks 15-17 (seeds 42, 123, 2026)
#   ETTm2         : tasks 18-20 (seeds 42, 123, 2026)
#   electricity   : tasks 21-23 (seeds 42, 123, 2026)
#   traffic       : tasks 24-26 (seeds 42, 123, 2026)
#
# Results layout (same as main jobs, merged by merge_results.py):
#   results/raw/long_term/lag_llama_pretrained_<DATASET>_s<SEED>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lag_llama_pretrained_lt.sh

#SBATCH --job-name lag-llama-lt
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-26%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lag_llama_lt_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lag_llama_lt_%A_%a.err

DATASETS=(
    illness       illness       illness
    exchange_rate exchange_rate exchange_rate
    ETTh1         ETTh1         ETTh1
    ETTh2         ETTh2         ETTh2
    weather       weather       weather
    ETTm1         ETTm1         ETTm1
    ETTm2         ETTm2         ETTm2
    electricity   electricity   electricity
    traffic       traffic       traffic
)
SEEDS=(
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
    42 123 2026
)

T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"
SEED="${SEEDS[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lag_llama_pretrained LT: $DATASET seed=$SEED — all 4 horizons (zero-shot)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"
echo "WARNING: autoregressive decoding — expect ~pred_len× longer inference time"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "models=[lag_llama_pretrained]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/lag_llama_pretrained_${DATASET}_s${SEED}" \
    "hydra.run.dir=outputs/lt/lag_llama_pretrained_${DATASET}_s${SEED}"

echo "[$SLURM_JOB_ID/$T] Finished: lag_llama_pretrained $DATASET seed=$SEED"
