#!/bin/bash
# Long-term benchmark — MEMORY-HEAVY models (contiformer, pathformer),
# pinned to the A100 node (talos, 40 GB), fine-grained per (dataset,seed,horizon).
#
# Both models OOM on the V100 nodes (32 GB) in fp32:
#   - contiformer: ODE solver retains a huge autograd graph; OOM'd at
#     batch 16/4/2 on V100 and at batch 4 on A100.  Now runs at batch=1
#     (override in model_configs) on the A100 only.
#   - pathformer: FFT encoder OOMs on the larger datasets (weather, electricity,
#     traffic) on V100; fits on the A100.  (Its already-completed small-dataset
#     cells are skipped by the incremental checkpoint.)
#
# Pinned to talos so both get the 40 GB card.  Dedicated *_mem_h* dirs; the 4
# memory-OK slow models run separately in run_lt_slow.sh.  merge_results.py
# dedups by run key, keeping the latest timestamp — this job runs after the
# earlier slow-group error rows, so its OK rows win.
#
# Grid: 9 datasets × 3 seeds × 4 horizons = 108 tasks.  illness uses horizons
# 24/36/48/60; every other dataset uses 96/192/336/720.  Resume skips cells
# already OK (e.g. pathformer on the small datasets), so most such tasks exit
# quickly.
#   index t: dataset = t/12, seed = (t/4)%3, horizon-slot = t%4
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>_mem_h<H>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_memheavy.sh

#SBATCH --job-name lt-memheavy
#SBATCH --partition dgx2
#SBATCH --nodelist=talos
#SBATCH --gres=gpu:1
#SBATCH --array=0-107%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_memheavy_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_memheavy_%A_%a.err

MEM_MODELS="contiformer,pathformer"

DATASETS=(illness exchange_rate ETTh1 ETTh2 weather ETTm1 ETTm2 electricity traffic)
SEEDS=(42 123 2026)
# Horizon slots: illness uses a different set; everything else the default.
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

# Heavy-channel datasets get a smaller global batch; per-model overrides in
# model_configs still cap contiformer to 1.
case "$DATASET" in
    traffic)     BS=8  ;;
    *)           BS=16 ;;
esac

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-memheavy: $DATASET seed=$SEED h=$HORIZON bs=$BS — contiformer,pathformer (A100)"
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
    "experiment.models=[$MEM_MODELS]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_mem_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_mem_h${HORIZON}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
