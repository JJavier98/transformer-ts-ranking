#!/bin/bash
# Smoke test for the long-term forecasting pipeline.
#
# Runs ALL eligible models on ETTh1 (all 4 horizons), 1 epoch, seed=42.
# Purpose: verify end-to-end pipeline correctness (model init → training step
# → evaluation step → metrics → parquet write) before launching the full
# experiment.  Expected runtime: 30–90 minutes on a V100.
#
# Usage:
#   sbatch scripts/sbatch/smoke_long_term.sh
#
# After completion, inspect results:
#   .venv/bin/python -c "
#   import pandas as pd
#   df = pd.read_parquet('results/smoke/long_term/ETTh1/results_raw.parquet')
#   print(df[['model_name','horizon','mae','error']].to_string())
#   print('OK:', df['error'].isna().sum(), '  FAIL:', df['error'].notna().sum())
#   "

#SBATCH --job-name smoke-lt
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/smoke_lt_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/smoke_lt_%j.err

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID] Smoke test — long-term / ETTh1 / all horizons / seed=42 / 1 epoch"
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
    experiment=long_term \
    "experiment.datasets=[ETTh1]" \
    wandb=disabled \
    "+results_dir=results/smoke/long_term/ETTh1" \
    "hydra.run.dir=outputs/smoke/lt_ETTh1"

echo "[$SLURM_JOB_ID] Smoke long-term finished."
