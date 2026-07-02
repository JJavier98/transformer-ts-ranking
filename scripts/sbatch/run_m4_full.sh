#!/bin/bash
# M4 short-term benchmark — all 6 frequency slices.
#
# Each task runs ALL eligible models × 3 seeds × 30 epochs on one M4
# frequency.  Results are saved per-frequency with no write conflicts
# between concurrently running tasks.
#
# Task layout (6 tasks total):
#   0  Yearly      pred_len=6
#   1  Quarterly   pred_len=8
#   2  Monthly     pred_len=18
#   3  Weekly      pred_len=13
#   4  Daily       pred_len=14
#   5  Hourly      pred_len=48
#
# Results layout:
#   results/raw/m4/<FREQ>/results_raw.parquet
# Merged with long-term results by scripts/sbatch/merge_results.py.
#
# Usage:
#   sbatch scripts/sbatch/run_m4_full.sh

#SBATCH --job-name m4-full
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/m4_full_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/m4_full_%A_%a.err

FREQUENCIES=(Yearly Quarterly Monthly Weekly Daily Hourly)
FREQ="${FREQUENCIES[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] M4 full: $FREQ — all models, seeds=[42,123,2026], 30 epochs"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    training.batch_size=16 \
    experiment=m4 \
    "experiment.datasets=[$FREQ]" \
    wandb=disabled \
    "+results_dir=results/raw/m4/$FREQ" \
    "hydra.run.dir=outputs/m4/$FREQ"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $FREQ (exit=$STATUS)"
exit $STATUS
