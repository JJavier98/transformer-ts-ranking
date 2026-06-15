#!/bin/bash
# SBATCH job array for the long-term forecasting benchmark.
# Each array task runs ONE dataset; all eligible models × 4 horizons × 3 seeds.
# Results land in results/raw/long_term/<DATASET>/ (separate dirs avoid write races).
#
# Usage:
#   sbatch scripts/sbatch/run_long_term.sh
#
# Datasets 0-6 run on the dgx partition. For electricity (idx 7) and traffic (idx 8)
# use run_long_term_heavy.sh with the dgx2 partition.

#SBATCH --job-name lt-bench
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-6%4
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/long_term_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/long_term_%A_%a.err

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Starting long-term benchmark for dataset: $DATASET"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "+results_dir=results/raw/long_term/$DATASET" \
    "hydra.run.dir=outputs/long_term/$DATASET"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
