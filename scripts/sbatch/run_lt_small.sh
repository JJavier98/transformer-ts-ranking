#!/bin/bash
# Long-term benchmark — small datasets (illness, exchange_rate, ETTh1, ETTh2).
#
# Each task runs ALL eligible models × all 4 horizons × 3 seeds × 30 epochs
# on one dataset.  These datasets are small enough that all seeds fit in one
# task well within the 4-day wall-time limit (estimated 10–45 h each).
#
# Eligible models come from configs/benchmark/model_capability_matrix.yaml.
# No experiment.models override: the capability matrix is the single source of
# truth.
#
# Results layout (one parquet per dataset, no write conflicts):
#   results/raw/long_term/<DATASET>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_small.sh

#SBATCH --job-name lt-small
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --time=2-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_small_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_small_%A_%a.err

DATASETS=(illness exchange_rate ETTh1 ETTh2)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] lt-small: $DATASET — all models, 4 horizons, seeds=[42,123,2026], 30 epochs"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/$DATASET" \
    "hydra.run.dir=outputs/lt/${DATASET}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
# Without this, a mid-run host-OOM kill leaves the shard silently incomplete
# (only the horizons processed before the kill) while SLURM reports success.
echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET (exit=$STATUS)"
exit $STATUS
