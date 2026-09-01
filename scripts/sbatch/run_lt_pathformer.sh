#!/bin/bash
# Long-term benchmark — pathformer, UNITARY tasks, only the datasets where it
# is incomplete (electricity, traffic, weather).
#
# pathformer completed exchange_rate, illness, ETTh1/h2, ETTm1/m2 (all 12 cells
# each).  It fails on:
#   - electricity (321ch) + traffic (862ch): OOM on the A100 at batch=8 (its FFT
#     encoder scales with channels).  Fix: batch=2 on these two datasets.
#   - weather: a ValueError on 4 cells (a numerical/distribution edge case, not
#     memory).  Re-run at the normal batch; cells that keep failing become N/A.
# Pinned to the A100 node (talos, 40GB) for the heavy-channel datasets.
#
# One (dataset, seed, horizon) cell per task.  Resume skips the 8 weather cells
# already OK.
#
# Grid: 3 datasets × 3 seeds × 4 horizons = 36 tasks (all use 96/192/336/720).
#   index t: dataset = t/12, seed = (t/4)%3, horizon = t%4
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>_path_h<H>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_pathformer.sh

#SBATCH --job-name lt-path
#SBATCH --partition dgx2
#SBATCH --nodelist=talos
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%6
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_path_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_path_%A_%a.err

DATASETS=(electricity traffic weather)
SEEDS=(42 123 2026)
HORIZONS=(96 192 336 720)

T=$SLURM_ARRAY_TASK_ID
DS_IDX=$(( T / 12 ))
SEED_IDX=$(( (T / 4) % 3 ))
H_IDX=$(( T % 4 ))

DATASET="${DATASETS[$DS_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
HORIZON="${HORIZONS[$H_IDX]}"

# electricity/traffic OOM at higher batch (many channels); weather is a retry
# of a non-memory error, so it keeps the normal batch.
case "$DATASET" in
    electricity|traffic) BS=2  ;;
    weather)             BS=16 ;;
esac

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-path: $DATASET seed=$SEED h=$HORIZON bs=$BS — pathformer (A100)"
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
    "experiment.models=[pathformer]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_path_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_path_h${HORIZON}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL surfaces as FAILED.
echo "[$SLURM_JOB_ID/$T] Finished: pathformer $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
