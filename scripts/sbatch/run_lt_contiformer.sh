#!/bin/bash
# Long-term benchmark — contiformer, UNITARY tasks (one cell each).
#
# contiformer OOM'd at batch 16/4/2/1 on both V100 and A100 with the full
# context (seq_len=96): its ODE solver retains a [B,T,T,D] tensor per function
# evaluation.  Last-resort fix: context_len=48 (set in model_configs; the ODE
# tensor scales with T^2 so this cuts memory ~4x) + batch=1.  With that the
# footprint is tiny, so tasks are NOT pinned to a node — they run anywhere.
#
# One (dataset, seed, horizon) cell per array task, so each job is short and
# schedules easily.  The incremental checkpoint skips any cell already OK.
#
# Grid: 9 datasets × 3 seeds × 4 horizons = 108 tasks.  illness uses horizons
# 24/36/48/60; every other dataset uses 96/192/336/720.
#   index t: dataset = t/12, seed = (t/4)%3, horizon-slot = t%4
#
# Results layout (unique dir per task -> no concurrent-write conflicts):
#   results/raw/long_term/<DATASET>_s<SEED>_conti_h<H>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_contiformer.sh

#SBATCH --job-name lt-conti
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-107%6
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_conti_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_conti_%A_%a.err

DATASETS=(illness exchange_rate ETTh1 ETTh2 weather ETTm1 ETTm2 electricity traffic)
SEEDS=(42 123 2026)
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

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-conti: $DATASET seed=$SEED h=$HORIZON — contiformer ctx=48 bs=1"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[contiformer]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_conti_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_conti_h${HORIZON}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL surfaces as FAILED.
echo "[$SLURM_JOB_ID/$T] Finished: contiformer $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
