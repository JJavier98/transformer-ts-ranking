#!/bin/bash
# ETTm1/ETTm2 slow follow-up — Batch 1: triformer, quatformer, lag_llama, spacetimeformer
#
# Run AFTER job 149281 (or any lt-bench job covering ETTm) has finished.
# Deduplication in the runner ensures no conflicts with previously written results.
#
# Estimated: ~70h per dataset (ETTm1 and ETTm2 run in parallel on separate nodes).
#
# Usage:
#   sbatch scripts/sbatch/run_ettm_slow_b1.sh

#SBATCH --job-name ettm-slow-b1
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/ettm_slow_b1_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/ettm_slow_b1_%A_%a.err

DATASETS=(ETTm1 ETTm2)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Ordered fast-to-slow within the batch to maximise results before any time limit.
MODELS="triformer,quatformer,lag_llama,spacetimeformer"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] ETTm slow follow-up B1: $DATASET | $MODELS"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[$MODELS]" \
    "+results_dir=results/raw/long_term/$DATASET" \
    "hydra.run.dir=outputs/long_term/${DATASET}_slow_b1"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
