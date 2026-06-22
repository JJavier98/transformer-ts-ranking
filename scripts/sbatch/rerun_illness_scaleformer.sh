#!/bin/bash
# Targeted re-run: scaleformer on illness (long-term).
#
# Fixes 6 failures from the initial illness run caused by the default
# scales=[8,4,2,1] not dividing illness horizons 36 and 60 (36%8!=0,
# 60%8!=0).  model_configs.py now sets scales=[1] in _MODEL_OVERRIDES;
# this job regenerates those 6 rows with the corrected config.
#
# merge_results.py deduplicates by (model_name, dataset_name, horizon,
# seed, task) keeping the latest run_timestamp, so these rows cleanly
# replace the error rows in the canonical parquet.
#
# Usage:
#   sbatch scripts/sbatch/rerun_illness_scaleformer.sh

#SBATCH --job-name lt-fix-illness
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/rerun_illness_scaleformer_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/rerun_illness_scaleformer_%j.err

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID] Rerun: scaleformer × illness — scales=[1] fix"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[illness]" \
    "models=[scaleformer]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/illness" \
    "hydra.run.dir=outputs/lt/illness_scaleformer_fix"

echo "[$SLURM_JOB_ID] Finished: scaleformer × illness rerun"
