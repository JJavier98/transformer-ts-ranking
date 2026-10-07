#!/bin/bash
# Batch-size invariance check for lag_llama_pretrained (zero-shot).
#
# Hypothesis: for lag_llama_pretrained the batch size is a pure throughput
# knob.  fit() is a no-op (zero-shot), the test DataLoader uses
# drop_last=False (identical set of evaluated windows), and every series is
# normalised independently (per-sample loc/scale), so predictions must match
# the canonical batch_size=16 run up to fp32 kernel noise.
#
# The autoregressive loop (pred_len steps × channels, no KV cache, a host
# sync per step) leaves the GPU almost idle, so a larger batch should cut
# wall time roughly proportionally.
#
# Reference: ETTh1 h96 seed 42 at batch_size=16 (existing row in
# results/raw/long_term/, ~1.5 h wall).  This job reruns the same cell at
# batch_size=256 and writes OUTSIDE results/raw/ so the merge never picks it
# up.  Compare MAE/MSE/RMSE against the reference row afterwards.
#
# Usage:
#   sbatch scripts/sbatch/validate_lag_llama_bs.sh

#SBATCH --job-name val-lag-bs
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/val_lag_bs_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/val_lag_bs_%j.err

BS=${BS:-256}

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID] lag_llama batch-size check: ETTh1 h96 seed=42 bs=$BS"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[42]" \
    "training.batch_size=$BS" \
    experiment=long_term \
    "experiment.datasets=[ETTh1]" \
    "experiment.models=[lag_llama_pretrained]" \
    "experiment.horizons=[96]" \
    wandb=disabled \
    "+results_dir=results/validation/lag_llama_bs${BS}_ETTh1_s42_h96" \
    "hydra.run.dir=outputs/validation/lag_llama_bs${BS}"
STATUS=$?

echo "[$SLURM_JOB_ID] Finished (exit=$STATUS)"
exit $STATUS
