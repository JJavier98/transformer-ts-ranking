#!/bin/bash
# M4 short-term benchmark — toto2 (Toto 2.0 zero-shot foundation model).
#
# toto2 has results on the long-term track but none on M4.  Like run_lt_toto2.sh
# it must run from the dedicated Python >=3.12 environment (.venv-toto2): its
# upstream package (toto-models) cannot be installed in the shared .venv.  It is
# zero-shot, so one task per frequency covers the 3 seeds (identical results for
# a frozen model, kept for protocol symmetry).
#
# Results layout:
#   results/raw/m4/<FREQ>_toto2/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_m4_toto2.sh

#SBATCH --job-name m4-toto2
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --time=1-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/m4_toto2_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/m4_toto2_%A_%a.err

FREQUENCIES=(Yearly Quarterly Monthly Weekly Daily Hourly)
FREQ="${FREQUENCIES[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
# NOTE: the Python 3.12 environment, NOT the shared .venv.
PYTHON="$REPO/.venv-toto2/bin/python"
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] M4 toto2: $FREQ — zero-shot, seeds 42/123/2026"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)  (must be >=3.12 for toto-models)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    training.batch_size=16 \
    experiment=m4 \
    "experiment.datasets=[$FREQ]" \
    "experiment.models=[toto2]" \
    wandb=disabled \
    "+results_dir=results/raw/m4/${FREQ}_toto2" \
    "hydra.run.dir=outputs/m4/${FREQ}_toto2"
STATUS=$?

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: toto2 $FREQ (exit=$STATUS)"
exit $STATUS
