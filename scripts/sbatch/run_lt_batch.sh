#!/bin/bash
# Generic long-term batch runner — runs a specific model subset on one dataset.
#
# Submit via sbatch --export; examples:
#
#   # All 3 seeds (default):
#   sbatch -J "lt-electricity-b1" \
#          -o logs/lt_electricity_b1_%j.out \
#          -e logs/lt_electricity_b1_%j.err \
#          --export=DATASET=electricity,MODELS="itransformer,patchtst,..." \
#          scripts/sbatch/run_lt_batch.sh
#
#   # Single seed (for slow models that exceed the 4-day limit with all seeds):
#   sbatch -J "lt-electricity-pathformer-s42" \
#          -o logs/lt_electricity_pathformer_s42_%j.out \
#          -e logs/lt_electricity_pathformer_s42_%j.err \
#          --export=DATASET=electricity,MODELS=pathformer,SEED=42 \
#          scripts/sbatch/run_lt_batch.sh
#
# Required env vars (set via --export):
#   DATASET  — one of: ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather electricity traffic
#   MODELS   — comma-separated model names in the order they should run (fastest first)
#
# Optional env vars:
#   SEED     — single integer (42 | 123 | 2026); omit to run all three seeds
#
# Results land in results/raw/long_term/$DATASET/; the runner deduplicates by
# (model, dataset, horizon, seed) so partial/per-seed runs compose cleanly.

#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

# Build the optional seed override — empty string when running all seeds.
if [[ -n "${SEED}" ]]; then
    SEED_ARG="training.seeds=[$SEED]"
    SEED_TAG=" seed=$SEED"
else
    SEED_ARG=""
    SEED_TAG=" seeds=[42,123,2026]"
fi

echo "[$SLURM_JOB_ID] Dataset: ${DATASET}  |  Models: ${MODELS}${SEED_TAG}"
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

echo "[$SLURM_JOB_ID] Finished: $DATASET ($MODELS${SEED_TAG})"
