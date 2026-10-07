#!/bin/bash
# Gap-fill of the SUPPLEMENTARY long-term datasets (electricity, traffic).
#
# One array task per row of configs/benchmark/fill_tasks_supplementary.tsv
# (built by scripts/make_fill_tasks.py from the persisted shards):
#   task_id  dataset  seed  horizon  batch_size  models
# Models that never ran share a task per cell; slow models and models retried
# after an out-of-memory error run alone, the latter with a smaller batch size.
#
# These cells are not part of the main ranking (decided 2026-10-05), so the
# array is submitted with a large --nice value to run after every core task:
#   sbatch --nice=5000 scripts/sbatch/run_fill_supplementary.sh
#
# Results layout (unique dir per task):
#   results/raw/long_term/<DATASET>_s<SEED>_suppfill<T>_h<H>/results_raw.parquet
# Resubmitting is safe: finished runs inside a task are skipped.

#SBATCH --job-name fill-supp
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-121
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/fill_supp_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/fill_supp_%A_%a.err

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"
TABLE="$REPO/configs/benchmark/fill_tasks_supplementary.tsv"
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

T=$SLURM_ARRAY_TASK_ID
# Row T of the table (line T+2: line 1 is the header).
IFS=$'\t' read -r TID DATASET SEED HORIZON BS MODELS <<< "$(sed -n "$((T + 2))p" "$TABLE")"
if [ "$TID" != "$T" ]; then
    echo "task table row mismatch: array index $T, table says '$TID'" >&2
    exit 2
fi

echo "[$SLURM_JOB_ID/$T] fill-supp: $DATASET seed=$SEED h=$HORIZON bs=$BS models=$MODELS"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    "training.batch_size=$BS" \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[$MODELS]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_suppfill${T}_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_suppfill${T}_h${HORIZON}"
STATUS=$?

echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
