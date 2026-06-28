#!/bin/bash
# Weather long-term rerun — seeds 42, 123, 2026.
#
# Replaces the original lt-medium array tasks for weather:
#   - weather_s42:  task 0 of 151198 timed out with no parquet (old code, no checkpoint)
#   - weather_s123: task 1 of 151198 cancelled (3d21h runtime, 78 combos in JSONL, no parquet)
#   - weather_s2026: task 2 of 151198 cancelled (hung 19h on spacetimeformer_h720)
#
# This script uses the current code with:
#   - Incremental checkpoint: parquet written after EVERY (model, horizon, seed) combo.
#   - bf16 autocast: enabled on A100/H100 for compatible models (14 models excluded).
#   - Batch-size overrides: spacetimeformer uses B=4 at ALL horizons.
#   - Context-length truncation: contiformer capped at T=96.
#   - TFT fix: 'tft' now in _SEQ2SEQ_MODELS (label_len passed correctly).
#   - lag_llama_pretrained fix: _build_lag_features monkey-patched (shape bug).
#
# Results write to the same dirs as the original tasks so the parquet shards
# merge correctly with the rest of the benchmark.
#
# Task layout:
#   0  weather  seed=42    → results/raw/long_term/weather_s42
#   1  weather  seed=123   → results/raw/long_term/weather_s123
#   2  weather  seed=2026  → results/raw/long_term/weather_s2026
#
# Estimated runtime: ~40–60 h per task (28 models × 4 horizons × 30 epochs,
# early-stopping typically cuts to 10–20 epochs for most models).
# Wall-time 4 days is a comfortable upper bound.
#
# Note on node selection: spacetimeformer_weather_h720 hung on dgx1 (V100) in
# the previous run (19h without completing one epoch).  If a task lands on dgx1
# and stalls again, cancel and resubmit with --nodelist=talos to force A100.
# Alternatively, use: sbatch --nodelist=talos run_lt_weather_rerun.sh
#
# Usage:
#   sbatch scripts/sbatch/run_lt_weather_rerun.sh

#SBATCH --job-name lt-weather
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_weather_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_weather_%A_%a.err

SEEDS=(42 123 2026)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] lt-weather: seed=$SEED — 28 models, 4 horizons, 30 epochs (incremental checkpoint)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[weather]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/weather_s${SEED}" \
    "hydra.run.dir=outputs/lt/weather_s${SEED}"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: weather seed=$SEED"
