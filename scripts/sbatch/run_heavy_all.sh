#!/bin/bash
# Single array covering ALL heavy dataset benchmark tasks in one sbatch call.
#
# 26 tasks total:
#   Tasks  0-13  Standard batches (B1-B7 × electricity + traffic, all 3 seeds)
#   Tasks 14-19  spacetimeformer  (3 seeds × electricity + traffic)
#   Tasks 20-25  pathformer       (3 seeds × electricity + traffic)
#
# Task layout:
#   task  0  electricity  B1  all seeds   ~50h
#   task  1  traffic      B1  all seeds   ~75h
#   task  2  electricity  B2  all seeds   ~57h
#   task  3  traffic      B2  all seeds   ~86h
#   task  4  electricity  B3  all seeds   ~58h
#   task  5  traffic      B3  all seeds   ~88h
#   task  6  electricity  B4  all seeds   ~53h
#   task  7  traffic      B4  all seeds   ~75h
#   task  8  electricity  B5  all seeds   ~40h
#   task  9  traffic      B5  all seeds   ~59h
#   task 10  electricity  B6  all seeds   ~40h
#   task 11  traffic      B6  all seeds   ~59h
#   task 12  electricity  B7  all seeds   ~42h
#   task 13  traffic      B7  all seeds   ~62h
#   task 14  electricity  spacetimeformer seed=42    ~75h
#   task 15  traffic      spacetimeformer seed=42    ~37h
#   task 16  electricity  spacetimeformer seed=123   ~75h
#   task 17  traffic      spacetimeformer seed=123   ~37h
#   task 18  electricity  spacetimeformer seed=2026  ~75h
#   task 19  traffic      spacetimeformer seed=2026  ~37h
#   task 20  electricity  pathformer      seed=42    ~70h
#   task 21  traffic      pathformer      seed=42   ~102h ⚠ borderline
#   task 22  electricity  pathformer      seed=123   ~70h
#   task 23  traffic      pathformer      seed=123  ~102h ⚠ borderline
#   task 24  electricity  pathformer      seed=2026  ~70h
#   task 25  traffic      pathformer      seed=2026 ~102h ⚠ borderline
#
# chronos2 and lag_llama (from-scratch) not included; use chronos_bolt and
# lag_llama_pretrained (pretrained, via run_foundation.sh) instead.
#
# Usage:
#   sbatch scripts/sbatch/run_heavy_all.sh

#SBATCH --job-name heavy-all
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-25%4
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/heavy_all_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/heavy_all_%A_%a.err

# ---------------------------------------------------------------------------
# Lookup tables  (indexed by SLURM_ARRAY_TASK_ID)
# ---------------------------------------------------------------------------

B1="itransformer,timexer,transformer,patchtst,multipatchformer,pyraformer,nonstationary_transformer,cats,reformer,tft,informer,fedformer"
B2="scaleformer,card,etsformer,triformer"
B3="basisformer,autoformer,airformer"
B4="quatformer,earthformer"
B5="contiformer"
B6="deformable_tst"
B7="crossformer"

DATASETS=(
    electricity traffic   # 0-1   B1
    electricity traffic   # 2-3   B2
    electricity traffic   # 4-5   B3
    electricity traffic   # 6-7   B4
    electricity traffic   # 8-9   B5
    electricity traffic   # 10-11 B6
    electricity traffic   # 12-13 B7
    electricity traffic   # 14-15 spacetimeformer s42
    electricity traffic   # 16-17 spacetimeformer s123
    electricity traffic   # 18-19 spacetimeformer s2026
    electricity traffic   # 20-21 pathformer s42
    electricity traffic   # 22-23 pathformer s123
    electricity traffic   # 24-25 pathformer s2026
)

MODELS_MAP=(
    "$B1" "$B1"
    "$B2" "$B2"
    "$B3" "$B3"
    "$B4" "$B4"
    "$B5" "$B5"
    "$B6" "$B6"
    "$B7" "$B7"
    "spacetimeformer" "spacetimeformer"
    "spacetimeformer" "spacetimeformer"
    "spacetimeformer" "spacetimeformer"
    "pathformer" "pathformer"
    "pathformer" "pathformer"
    "pathformer" "pathformer"
)

SEEDS=(
    "" "" "" "" "" "" "" "" "" "" "" "" "" ""  # 0-13: all seeds
    "42"   "42"    # 14-15
    "123"  "123"   # 16-17
    "2026" "2026"  # 18-19
    "42"   "42"    # 20-21
    "123"  "123"   # 22-23
    "2026" "2026"  # 24-25
)

# ---------------------------------------------------------------------------
# Resolve parameters for this task
# ---------------------------------------------------------------------------

T=$SLURM_ARRAY_TASK_ID
DATASET="${DATASETS[$T]}"
MODELS="${MODELS_MAP[$T]}"
SEED="${SEEDS[$T]}"

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

if [[ -n "$SEED" ]]; then
    SEED_ARG="training.seeds=[$SEED]"
    SEED_TAG=" seed=$SEED"
else
    SEED_ARG=""
    SEED_TAG=" seeds=[42,123,2026]"
fi

echo "[$SLURM_JOB_ID/$T] $DATASET | $MODELS${SEED_TAG}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    experiment=long_term \
    "experiment.datasets=[$DATASET]" \
    "experiment.models=[$MODELS]" \
    ${SEED_ARG:+"$SEED_ARG"} \
    "+results_dir=results/raw/long_term/$DATASET" \
    "hydra.run.dir=outputs/long_term/${DATASET}_batch"

echo "[$SLURM_JOB_ID/$T] Finished: $DATASET"
