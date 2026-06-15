#!/bin/bash
# Array job for the two heavy long-term datasets (electricity + traffic).
# Task 0 = electricity (321 channels), task 1 = traffic (862 channels).
#
# Submit via sbatch --export; examples:
#
#   # All 3 seeds:
#   sbatch -J "heavy-b1" \
#          -o logs/heavy_b1_%A_%a.out \
#          -e logs/heavy_b1_%A_%a.err \
#          --export=MODELS="itransformer,patchtst,..." \
#          scripts/sbatch/run_heavy_batch.sh
#
#   # Single seed (for slow models — pathformer, spacetimeformer):
#   sbatch -J "heavy-pathformer-s42" \
#          -o logs/heavy_pathformer_s42_%A_%a.out \
#          -e logs/heavy_pathformer_s42_%A_%a.err \
#          --export=MODELS=pathformer,SEED=42 \
#          scripts/sbatch/run_heavy_batch.sh
#
# Required env vars (via --export):
#   MODELS  — comma-separated model names, ordered fastest-first
#
# Optional env vars:
#   SEED    — single integer (42 | 123 | 2026); omit to run all three seeds
#
# Results land in results/raw/long_term/{electricity,traffic}/.
# The runner deduplicates by (model, dataset, horizon, seed) so per-seed
# jobs and multi-seed jobs compose correctly.

#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2

DATASETS=(electricity traffic)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

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

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] $DATASET | $MODELS${SEED_TAG}"
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
    "hydra.run.dir=outputs/long_term/${DATASET}_batch"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
