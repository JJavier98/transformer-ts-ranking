#!/bin/bash
# Long-term benchmark — medium datasets (weather, ETTm1, ETTm2), split by seed.
#
# Each task runs ALL eligible models × all 4 horizons × ONE seed × 30 epochs
# on one dataset.  Splitting by seed keeps each task under ~65 h (< 4-day
# wall-time limit).  Separate results dirs prevent write conflicts between
# concurrently running tasks.
#
# Task layout (9 tasks total):
#   0  weather    seed=42
#   1  weather    seed=123
#   2  weather    seed=2026
#   3  ETTm1      seed=42
#   4  ETTm1      seed=123
#   5  ETTm1      seed=2026
#   6  ETTm2      seed=42
#   7  ETTm2      seed=123
#   8  ETTm2      seed=2026
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>/results_raw.parquet
# These shards are merged by scripts/sbatch/merge_results.py after all tasks
# complete.
#
# Usage:
#   sbatch scripts/sbatch/run_lt_medium.sh

#SBATCH --job-name lt-medium
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_medium_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_medium_%A_%a.err

DATASETS=(weather weather weather ETTm1 ETTm1 ETTm1 ETTm2 ETTm2 ETTm2)
SEEDS=(42 123 2026 42 123 2026 42 123 2026)

T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"
SEED="${SEEDS[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-medium: $DATASET seed=$SEED — all models, 4 horizons, 30 epochs"
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
