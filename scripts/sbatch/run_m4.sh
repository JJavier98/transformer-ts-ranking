#!/bin/bash
# SBATCH job array for the M4 short-term benchmark.
# Each array task runs ONE M4 frequency slice; all eligible models × 3 seeds.
# Results land in results/raw/m4/<FREQUENCY>/.
#
# Usage:
#   sbatch scripts/sbatch/run_m4.sh

#SBATCH --job-name m4-bench
#SBATCH --partition dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%4
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/m4_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/m4_%A_%a.err

FREQUENCIES=(Yearly Quarterly Monthly Weekly Daily Hourly)
FREQ="${FREQUENCIES[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate torch_env

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Starting M4 benchmark for frequency: $FREQ"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

cd "$REPO"
mkdir -p "$REPO/logs"

python "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=m4 \
    "experiment.datasets=[$FREQ]" \
    "+results_dir=results/raw/m4/$FREQ" \
    "hydra.run.dir=outputs/m4/$FREQ"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $FREQ"
