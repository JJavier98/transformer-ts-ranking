#!/bin/bash
# Long-term benchmark — SLOW models, fine-grained per (dataset, seed, horizon).
#
# The 6 slow models (pathformer, triformer, quatformer, spacetimeformer,
# deformable_tst, contiformer) are 5–15× slower per cell than the other 22.
# A single one — pathformer on weather h96 — takes ~15 h.  Running them inside
# the per-seed medium/heavy jobs blew the 4-day wall-time (those jobs only
# reached horizons 96/192 before TIMEOUT).  Here each task runs the 6 slow
# models on ONE (dataset, seed, horizon), which always fits well within 4 days.
#
# The fast 22 models are handled by run_lt_medium.sh / run_lt_heavy.sh.
# Results go to dedicated dirs; merge_results.py deduplicates by run key, so any
# overlap with slow-model rows already present in the per-seed dirs collapses.
#
# Grid: 5 datasets × 3 seeds × 4 horizons = 60 tasks.
#   datasets: weather, ETTm1, ETTm2 (medium) + electricity, traffic (heavy)
#   index t: dataset = t/12, seed = (t/4)%3, horizon = t%4
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>_slow_h<H>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_slow.sh

#SBATCH --job-name lt-slow
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-59%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_slow_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_slow_%A_%a.err

# 4 slow-but-memory-OK models — run on any node (V100 or A100).
# contiformer and pathformer OOM on V100 and are handled separately by
# run_lt_memheavy.sh (pinned to the A100 node).
SLOW_MODELS="triformer,quatformer,spacetimeformer,deformable_tst"

DATASETS=(weather ETTm1 ETTm2 electricity traffic)
SEEDS=(42 123 2026)
HORIZONS=(96 192 336 720)

T=$SLURM_ARRAY_TASK_ID
DS_IDX=$(( T / 12 ))
SEED_IDX=$(( (T / 4) % 3 ))
H_IDX=$(( T % 4 ))

DATASET="${DATASETS[$DS_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
HORIZON="${HORIZONS[$H_IDX]}"

# Heavy datasets need a smaller global batch (many channels); the per-model
# overrides in model_configs cap spacetimeformer/contiformer to 4 regardless.
case "$DATASET" in
    traffic)     BS=8  ;;
    electricity) BS=16 ;;
    *)           BS=16 ;;
esac

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-slow: $DATASET seed=$SEED h=$HORIZON bs=$BS — 4 slow models, 30 epochs"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    "training.batch_size=$BS" \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[$SLOW_MODELS]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_slow_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_slow_h${HORIZON}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
