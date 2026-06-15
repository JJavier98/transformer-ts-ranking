#!/bin/bash
# SBATCH job array for the foundation-model baseline track.
# Runs chronos_bolt and lag_llama_pretrained (both zero-shot, official
# pretrained weights from HuggingFace) on all 9 long-term datasets.
#
# Foundation models skip training (fit() is a no-op) so each task is faster
# than a from-scratch run. One GPU is requested for inference only.
#
# Usage:
#   sbatch scripts/sbatch/run_foundation.sh
#
# Results land in results/raw/foundation/<DATASET>/ separate from the
# from-scratch long-term results.

#SBATCH --job-name fm-bench
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%4
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/foundation_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/foundation_%A_%a.err

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather electricity traffic)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

# Cache HuggingFace weights in the project dir to avoid re-downloading per node.
export HF_HOME="$REPO/.hf_cache"
mkdir -p "$HF_HOME"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Foundation benchmark: $DATASET"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[chronos_bolt,lag_llama_pretrained]" \
    "+results_dir=results/raw/foundation/$DATASET" \
    "hydra.run.dir=outputs/foundation/$DATASET"

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Finished: $DATASET"
