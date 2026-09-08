#!/bin/bash
# Long-term benchmark — lag_llama_pretrained, UNITARY tasks (one cell each).
#
# lag_llama is autoregressive: it runs pred_len sequential forward passes, so it
# is ~pred_len x slower than the parallel models. Sharing a task with the other
# models made it monopolise 3 GPUs for 3 days without producing a single row
# (it blocked the 22 fast models at long horizons on electricity). It therefore
# runs isolated, one (dataset, seed, horizon) cell per task.
#
# IMPORTANT: resume does NOT help *within* a cell — the autoregressive rollout is
# not checkpointed mid-cell, so a cell either finishes inside the 4-day wall-time
# or never does. Indices are ordered COST-ASCENDING (cheap datasets and short
# horizons first, traffic/h720 last) so the tractable cells land early; any cell
# that still exceeds the wall-time is marked N/A with that justification.
#
# Grid: 9 datasets x 3 seeds x 4 horizons = 108 tasks. illness uses horizons
# 24/36/48/60; every other dataset uses 96/192/336/720.
#   index t: dataset = t/12, seed = (t/4)%3, horizon-slot = t%4
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>_lag_h<H>/results_raw.parquet
#
# Usage (normally via scripts/sbatch/submit_resume.sh, which gives it the
# highest nice so it runs dead last):
#   sbatch scripts/sbatch/run_lt_lag_llama.sh

#SBATCH --job-name lt-lagllama
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-107%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_lag_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_lag_%A_%a.err

# Cost-ascending: cheapest datasets first, the 862-channel traffic last.
DATASETS=(illness exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic)
SEEDS=(42 123 2026)
# Horizons ascending == cost ascending for an autoregressive model.
DEFAULT_H=(96 192 336 720)
ILLNESS_H=(24 36 48 60)

T=$SLURM_ARRAY_TASK_ID
DS_IDX=$(( T / 12 ))
SEED_IDX=$(( (T / 4) % 3 ))
H_IDX=$(( T % 4 ))

DATASET="${DATASETS[$DS_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
if [ "$DATASET" = "illness" ]; then
    HORIZON="${ILLNESS_H[$H_IDX]}"
else
    HORIZON="${DEFAULT_H[$H_IDX]}"
fi

case "$DATASET" in
    traffic)     BS=4 ;;
    electricity) BS=8 ;;
    *)           BS=16 ;;
esac

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-lagllama: $DATASET seed=$SEED h=$HORIZON bs=$BS (autoregressive)"
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
    "experiment.models=[lag_llama_pretrained]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_lag_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_lag_h${HORIZON}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL surfaces as FAILED.
echo "[$SLURM_JOB_ID/$T] Finished: lag_llama $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
