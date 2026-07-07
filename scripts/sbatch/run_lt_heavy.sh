#!/bin/bash
# Long-term benchmark — heavy datasets (electricity, traffic), split by seed.
#
# Each task runs ALL eligible models × all 4 horizons × ONE seed × 30 epochs.
# Splitting by seed keeps each task under ~80 h for electricity and ~90 h for
# traffic (both under the 4-day wall-time limit).
#
# batch_size is set conservatively to prevent OOM:
#   electricity (321 channels): batch_size=16
#   traffic     (862 channels): batch_size=8
#
# Task layout (6 tasks total):
#   0  electricity  seed=42
#   1  electricity  seed=123
#   2  electricity  seed=2026
#   3  traffic      seed=42
#   4  traffic      seed=123
#   5  traffic      seed=2026
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>/results_raw.parquet
# Merged by scripts/sbatch/merge_results.py after completion.
#
# Usage:
#   sbatch scripts/sbatch/run_lt_heavy.sh

#SBATCH --job-name lt-heavy
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_heavy_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_heavy_%A_%a.err

DATASETS=(electricity electricity electricity traffic traffic traffic)
SEEDS=(42 123 2026 42 123 2026)
BATCH_SIZES=(16 16 16 8 8 8)

T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"
SEED="${SEEDS[$T]}"
BS="${BATCH_SIZES[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

# Fast models only (22).  The 6 slow models run in run_lt_slow.sh (fine-grained
# per-(dataset,seed,horizon)) so they don't blow the 4-day wall-time on the
# large heavy datasets (electricity 321ch, traffic 862ch).
FAST_MODELS="airformer,autoformer,basisformer,card,cats,chronos_bolt,crossformer,earthformer,etsformer,fedformer,informer,itransformer,lag_llama_pretrained,multipatchformer,nonstationary_transformer,patchtst,pyraformer,reformer,scaleformer,tft,timexer,transformer"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-heavy: $DATASET seed=$SEED batch_size=$BS — 22 fast models, 4 horizons, 30 epochs"
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
    "experiment.models=[$FAST_MODELS]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED (exit=$STATUS)"
exit $STATUS
