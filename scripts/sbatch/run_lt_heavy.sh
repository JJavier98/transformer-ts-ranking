#!/bin/bash
# Long-term benchmark — HEAVY datasets (electricity, traffic), FAST models,
# fine-grained per (dataset, seed, horizon).
#
# electricity (321 channels) and traffic (862 channels) are so large that even
# the 22 fast models did not fit the 4-day wall-time per seed: per-seed tasks
# TIMEOUT'd having reached only horizons 96/192.  Here each task runs the 22
# fast models on ONE (dataset, seed, horizon).  electricity fits comfortably;
# traffic may still need ~2 resume rounds per cell (862 channels in fp32), but
# the incremental checkpoint makes each round advance and eventually complete.
#
# The 6 slow models on these datasets are handled by run_lt_slow.sh
# (per-(dataset,seed,horizon), *_slow_h* dirs).  merge_results.py dedups by run
# key so all shards merge cleanly.
#
# Grid: 2 datasets × 3 seeds × 4 horizons = 24 tasks.
#   index t: dataset = t/12, seed = (t/4)%3, horizon = t%4
#
# Results layout:
#   results/raw/long_term/<DATASET>_s<SEED>_h<H>/results_raw.parquet
#
# Usage:
#   sbatch scripts/sbatch/run_lt_heavy.sh

#SBATCH --job-name lt-heavy
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-23%4
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_heavy_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_heavy_%A_%a.err

# Fast models only (22).  Slow models -> run_lt_slow.sh.
FAST_MODELS="airformer,autoformer,basisformer,card,cats,chronos_bolt,crossformer,earthformer,etsformer,fedformer,informer,itransformer,lag_llama_pretrained,multipatchformer,nonstationary_transformer,patchtst,pyraformer,reformer,scaleformer,tft,timexer,transformer"

DATASETS=(electricity traffic)
SEEDS=(42 123 2026)
HORIZONS=(96 192 336 720)

T=$SLURM_ARRAY_TASK_ID
DS_IDX=$(( T / 12 ))
SEED_IDX=$(( (T / 4) % 3 ))
H_IDX=$(( T % 4 ))

DATASET="${DATASETS[$DS_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
HORIZON="${HORIZONS[$H_IDX]}"

case "$DATASET" in
    traffic)     BS=8  ;;
    electricity) BS=16 ;;
    *)           BS=16 ;;
esac

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] lt-heavy: $DATASET seed=$SEED h=$HORIZON bs=$BS — 22 fast models, 30 epochs"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[$SEED]" \
    "training.batch_size=$BS" \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[$FAST_MODELS]" \
    "experiment.horizons=[$HORIZON]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/${DATASET}_s${SEED}_h${HORIZON}" \
    "hydra.run.dir=outputs/lt/${DATASET}_s${SEED}_h${HORIZON}"
STATUS=$?

# Propagate the Python exit code so a crash or SIGKILL (137) surfaces as a
# FAILED job in sacct instead of being masked as COMPLETED by a trailing echo.
echo "[$SLURM_JOB_ID/$T] Finished: $DATASET seed=$SEED h=$HORIZON (exit=$STATUS)"
exit $STATUS
