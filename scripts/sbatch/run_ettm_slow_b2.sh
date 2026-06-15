#!/bin/bash
# ETTm1/ETTm2 slow follow-up — Batch 2: chronos2
#
# ~21h per dataset (all seeds). Safe to run as a single job.
# Supports optional SEED env var for further fragmentation.
#
# Usage:
#   sbatch scripts/sbatch/run_ettm_slow_b2.sh

#SBATCH --job-name ettm-slow-b2
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/ettm_slow_b2_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/ettm_slow_b2_%A_%a.err

DATASETS=(ETTm1 ETTm2)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

MODELS="chronos2"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

if [[ -n "${SEED}" ]]; then
    SEED_ARG="training.seeds=[$SEED]"
    SEED_TAG=" seed=$SEED"
else
    SEED_ARG=""
    SEED_TAG=" seeds=[42,123,2026]"
fi

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] ETTm slow B2 (chronos2): $DATASET${SEED_TAG}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[$MODELS]" \
    ${SEED_ARG:+"$SEED_ARG"} \
    "+results_dir=results/raw/long_term/$DATASET" \
    "hydra.run.dir=outputs/long_term/${DATASET}_slow_b2"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
