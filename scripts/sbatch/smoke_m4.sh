#!/bin/bash
# Smoke test for the M4 short-term forecasting pipeline.
#
# Runs ALL eligible models on the Monthly frequency slice, 1 epoch, seed=42.
# Purpose: verify M4 end-to-end correctness (M4SeriesDataset → training →
# OWA evaluation → parquet write) before launching the full experiment.
# Expected runtime: 30–90 minutes on a V100.
#
# Usage:
#   sbatch scripts/sbatch/smoke_m4.sh
#
# After completion, inspect results:
#   .venv/bin/python -c "
#   import pandas as pd
#   df = pd.read_parquet('results/smoke/m4/Monthly/results_raw.parquet')
#   print(df[['model_name','owa','smape','mase','error']].to_string())
#   print('OK:', df['error'].isna().sum(), '  FAIL:', df['error'].notna().sum())
#   "

#SBATCH --job-name smoke-m4
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/smoke_m4_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/smoke_m4_%j.err

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID] Smoke test — M4 / Monthly / seed=42 / 1 epoch"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    training.epochs=1 \
    "training.seeds=[42]" \
    training.batch_size=16 \
    training.patience=1 \
    experiment=m4 \
    "experiment.datasets=[Monthly]" \
    wandb=disabled \
    "+results_dir=results/smoke/m4/Monthly" \
    "hydra.run.dir=outputs/smoke/m4_Monthly"

echo "[$SLURM_JOB_ID] Smoke M4 finished."
