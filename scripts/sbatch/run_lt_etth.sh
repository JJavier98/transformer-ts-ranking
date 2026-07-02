#!/bin/bash
# Long-term benchmark — ETTh1 / ETTh2, split by seed.
#
# Per-seed variant (replaces the combined-seed run in run_lt_small.sh for the
# ETTh datasets).  Splitting by seed keeps each task short so a host-OOM kill
# on a shared node loses at most one seed's remaining horizons instead of a
# 1.5-day combined run — the failure mode that left the combined ETTh shards
# stuck at horizons 96/192.  Each task runs ALL eligible models × 4 horizons ×
# ONE seed × 30 epochs.
#
# The existing combined ETTh1/ETTh2 results were split into these same per-seed
# dirs, so the incremental checkpoint resumes: horizons 96/192 are skipped and
# only 336/720 (plus the contiformer cells, now fixed with batch_size=4) run.
#
# Task layout (6 tasks):
#   0 ETTh1 seed=42     3 ETTh2 seed=42
#   1 ETTh1 seed=123    4 ETTh2 seed=123
#   2 ETTh1 seed=2026   5 ETTh2 seed=2026
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>/results_raw.parquet
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
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED (exit=$STATUS)"
exit $STATUS
