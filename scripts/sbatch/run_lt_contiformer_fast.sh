#!/bin/bash
# Long-term benchmark — contiformer_fast, UNITARY tasks (one cell each).
#
# contiformer_fast is the S-TransformerTS variant of ContiFormer with one ODE
# trajectory per start point, replayed from CUDA graphs (library branch
# feat/contiformer-ode-checkpoint).  It runs with the canonical settings of
# every other model: seq_len=96 (no context override), batch_size=16,
# paper_ready preset, fp32.  Cost probe (V100): ~1.2 s per training step,
# ~0.2-0.8 h per epoch depending on the dataset (artifacts/review/
# contiformer_probe_v4_cuda_graphs.json).
#
# Scope: the 7 core datasets of the main ranking.
# Grid: 7 datasets × 3 seeds × 4 horizons = 84 tasks, cost-ascending
# (smallest datasets first).  illness uses horizons 24/36/48/60; every other
# dataset uses 96/192/336/720.
#   index t: dataset = t/12, seed = (t/4)%3, horizon-slot = t%4
#
# Results layout (unique dir per task -> no concurrent-write conflicts):
#   results/raw/long_term/<DATASET>_s<SEED>_cfast_h<H>/results_raw.parquet
#
# Usage:
#   sbatch --array=0 scripts/sbatch/run_lt_contiformer_fast.sh      # smoke cell
#   sbatch scripts/sbatch/run_lt_contiformer_fast.sh               # full grid

#SBATCH --job-name lt-cfast
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-83
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_cfast_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_cfast_%A_%a.err

DATASETS=(illness exchange_rate ETTh1 ETTh2 ETTm1 ETTm2 weather)
SEEDS=(42 123 2026)
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

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-cfast: $DATASET seed=$SEED h=$HORIZON — contiformer_fast bs=16"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Library: $(git -C "$REPO/s-transformers-lib" rev-parse --short HEAD)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[contiformer_fast]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_cfast_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_cfast_h${HORIZON}"
STATUS=$?

echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
