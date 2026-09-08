#!/bin/bash
# Long-term benchmark — toto2 (Toto 2.0 zero-shot foundation model).
#
# toto2 is the ONLY model that cannot run in the shared .venv: its upstream
# package (toto-models) requires Python >=3.12 while the benchmark environment
# is 3.11. It therefore runs from the dedicated .venv-toto2 interpreter.
# See the library README ("Toto 2.0 environment setup") and the capability
# matrix entry for toto2.
#
# It is zero-shot (fit() is a no-op, inference only), so it is very cheap:
# a full cell takes seconds. One task per dataset covers all 4 horizons x 3
# seeds (12 cells) comfortably. Seeds give identical results for a frozen
# model — kept for protocol symmetry with the other foundation models.
#
# Results layout (own dirs; merge_results dedups by run key):
#   results/raw/long_term/<DATASET>_toto2/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_toto2.sh

#SBATCH --job-name lt-toto2
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%4
#SBATCH --time=1-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_toto2_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_toto2_%A_%a.err

DATASETS=(illness exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic)
T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"

# Heavy-channel datasets get a smaller batch.
case "$DATASET" in
    traffic)     BS=4 ;;
    electricity) BS=8 ;;
    *)           BS=16 ;;
esac

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
# NOTE: the Python 3.12 environment, NOT the shared .venv.
PYTHON="$REPO/.venv-toto2/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-toto2: $DATASET bs=$BS — zero-shot, 4 horizons x 3 seeds"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)  (must be >=3.12 for toto-models)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.batch_size=$BS" \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[toto2]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_toto2" \
    "hydra.run.dir=outputs/lt/${DATASET}_toto2"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL surfaces as FAILED.
echo "[$SLURM_JOB_ID/$T] Finished: toto2 $DATASET (exit=$STATUS)"
exit $STATUS
