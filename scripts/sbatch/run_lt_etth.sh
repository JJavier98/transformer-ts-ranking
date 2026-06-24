#!/bin/bash
# Long-term benchmark — ETTh1 and ETTh2, split by seed.
#
# Replaces the failed runs from 151197_2 and 151197_3 which timed out at
# the 2-day limit.  Each task runs ONE dataset × ONE seed × all 4 horizons
# × 30 epochs within the 4-day wall-time limit.
#
# The runner now writes the parquet incrementally after every combo and
# skips already-completed combos on re-entry (checkpoint/resume).
#
# Task layout (6 tasks total):
#   0  ETTh1  seed=42
#   1  ETTh1  seed=123
#   2  ETTh1  seed=2026
#   3  ETTh2  seed=42
#   4  ETTh2  seed=123
#   5  ETTh2  seed=2026
#
# Results layout (merged by merge_results.py):
#   results/raw/long_term/ETTh1_s<SEED>/results_raw.parquet
#   results/raw/long_term/ETTh2_s<SEED>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_etth.sh

#SBATCH --job-name lt-etth
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_etth_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_etth_%A_%a.err

DATASETS=(ETTh1 ETTh1 ETTh1 ETTh2 ETTh2 ETTh2)
SEEDS=(42 123 2026 42 123 2026)

T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"
SEED="${SEEDS[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-etth: $DATASET seed=$SEED — all models, 4 horizons, 30 epochs"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}"

echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED"
