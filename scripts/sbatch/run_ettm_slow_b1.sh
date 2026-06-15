#!/bin/bash
# ETTm1/ETTm2 slow follow-up — Batch 1: triformer, quatformer, spacetimeformer
# (lag_llama from-scratch excluded; only lag_llama_pretrained is in scope)
#
# Run AFTER job 149281 (lt-bench covering ETTm) has finished.
# Deduplication ensures no conflicts with previously written results.
#
# Estimated: ~57h per dataset (all seeds). For shorter wall-clock, run per seed (~19h):
#
# Usage (all seeds, ~57h):
#   sbatch scripts/sbatch/run_ettm_slow_b1.sh
#
# Usage (per seed, ~19h each):
#   for s in 42 123 2026; do
#       sbatch -J "ettm-slow-b1-s$s" --export=SEED=$s scripts/sbatch/run_ettm_slow_b1.sh
#   done

#SBATCH --job-name ettm-slow-b1
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/ettm_slow_b1_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/ettm_slow_b1_%A_%a.err

DATASETS=(ETTm1 ETTm2)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Ordered fast-to-slow so more models complete if the job hits the wall.
# lag_llama (from-scratch) excluded — only lag_llama_pretrained is in scope.
MODELS="triformer,quatformer,spacetimeformer"

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

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] ETTm slow B1: $DATASET | $MODELS${SEED_TAG}"
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
    "hydra.run.dir=outputs/long_term/${DATASET}_slow_b1"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
