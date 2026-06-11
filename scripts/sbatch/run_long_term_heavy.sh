#!/bin/bash
# SBATCH job array for the heavy long-term datasets (electricity, traffic).
# These have 321 and 862 channels respectively and need the dgx2 partition.
# Results land in results/raw/long_term/<DATASET>/.
#
# Usage:
#   sbatch scripts/sbatch/run_long_term_heavy.sh

#SBATCH --job-name lt-heavy
#SBATCH --partition dgx2
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/long_term_heavy_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/long_term_heavy_%A_%a.err

DATASETS=(electricity traffic)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate torch_env

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Starting heavy long-term benchmark for dataset: $DATASET"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

cd "$REPO"
mkdir -p "$REPO/logs"

python "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "+results_dir=results/raw/long_term/$DATASET" \
    "hydra.run.dir=outputs/long_term/$DATASET"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
