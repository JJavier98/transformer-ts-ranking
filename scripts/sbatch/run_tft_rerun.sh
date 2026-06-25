#!/bin/bash
# TFT-only rerun for shards corrupted by the _SEQ2SEQ_MODELS bug.
#
# Background: 'tft' was missing from _SEQ2SEQ_MODELS (fixed in commit 1e7d1a9).
# With label_len=0, Python's x[:, -0:, :] returns the full encoder sequence
# instead of the last 48 steps, silently corrupting TFT decoder context.
#
# WORKFLOW — run these steps in order:
#   1. Wait for jobs 151198_1 (weather_s123), 151198_2 (weather_s2026),
#      151198_3 (ETTm1_s42), 151198_5 (ETTm1_s2026) to finish.
#      Check: squeue -u $USER
#   2. Purge wrong TFT rows from those shards:
#      .venv/bin/python scripts/sbatch/cleanup_tft_rows.py
#   3. Then submit this script:
#      sbatch scripts/sbatch/run_tft_rerun.sh
#
# Note: exchange_rate and illness were already cleaned immediately after the
# bug was found (12 rows each).  This script covers ALL affected shards so a
# single submission is enough — incremental checkpoint skips already-correct rows.
#
# Task layout (10 tasks — one per dataset × seed shard):
#   0  exchange_rate   seeds=[42,123,2026]   dir: long_term/exchange_rate
#   1  illness         seeds=[42,123,2026]   dir: long_term/illness
#   2  weather         seed=123              dir: long_term/weather_s123
#   3  weather         seed=2026             dir: long_term/weather_s2026
#   4  ETTm1           seed=42               dir: long_term/ETTm1_s42
#   5  ETTm1           seed=2026             dir: long_term/ETTm1_s2026
#   6  weather         seed=42               dir: long_term/weather_s42
#   7  ETTm2           seed=42               dir: long_term/ETTm2_s42
#   8  ETTm2           seed=123              dir: long_term/ETTm2_s123
#   9  ETTm2           seed=2026             dir: long_term/ETTm2_s2026
#
# Tasks 6–9 are defensive: those shards are produced by pending jobs that use
# the fixed code, so their TFT rows will already be correct.  The incremental
# checkpoint skips them at no cost; they are included so this job is complete
# as a standalone TFT coverage guarantee across all 9 long-term datasets.
#
# Each TFT run: 4 horizons × ~10–30 min = ~2 h per task.
# 8-hour wall-time is a comfortable upper bound.

#SBATCH --job-name tft-rerun
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%4
#SBATCH --time=0-08:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/tft_rerun_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/tft_rerun_%A_%a.err

# Parallel arrays: one entry per task index
DATASETS=(exchange_rate illness weather      weather       ETTm1      ETTm1        weather     ETTm2      ETTm2       ETTm2)
SEEDS=(   ""            ""       "123"       "2026"        "42"       "2026"       "42"        "42"       "123"       "2026")
DIRS=(    exchange_rate illness  weather_s123 weather_s2026 ETTm1_s42 ETTm1_s2026  weather_s42 ETTm2_s42  ETTm2_s123  ETTm2_s2026)

T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"
SEED="${SEEDS[$T]}"
RESDIR="${DIRS[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] tft-rerun: $DATASET seed='${SEED:-all}' → results/raw/long_term/$RESDIR"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

# Build args array; add seed override only when a specific seed is requested.
ARGS=(
    "$REPO/scripts/launch_benchmark.py"
    training=paper_ready
    training.batch_size=16
    experiment=long_term
    "experiment.datasets=[$DATASET]"
    "experiment.models=[tft]"
    wandb=disabled
    "+results_dir=results/raw/long_term/$RESDIR"
    "hydra.run.dir=outputs/tft_rerun/$RESDIR"
)

if [ -n "$SEED" ]; then
    ARGS+=("training.seeds=[$SEED]")
fi

"$PYTHON" "${ARGS[@]}"

echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed='${SEED:-all}'"
