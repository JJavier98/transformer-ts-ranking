#!/bin/bash
# Submit slow-model jobs that were excluded from the main batches because they
# exceed 4 days when running all 3 seeds together.
#
# chronos2 and lag_llama (from-scratch) are NOT included — the benchmark uses
# only their pretrained versions (chronos_bolt, lag_llama_pretrained).
#
# Strategy: one job per (model, dataset, seed) — each fits well within 96h.
#
# Estimated wall-clock per job (single seed, 4 horizons):
#
#   pathformer      on electricity:  ~70h  (209h ÷ 3)  ✅
#   spacetimeformer on traffic:      ~37h  (110h ÷ 3)  ✅
#   pathformer      on traffic:     ~102h  (307h ÷ 3)  ⚠ borderline — submitted anyway;
#                                                        will accumulate partial horizons
#                                                        if the job hits the wall.
#
# Usage:
#   bash scripts/sbatch/submit_slow_seeded.sh

set -e
REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
SCRIPT="$REPO/scripts/sbatch/run_lt_batch.sh"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

submit_seeded() {
    local model="$1" dataset="$2" seed="$3" est="$4"
    local tag="${model}_${dataset}_s${seed}"
    local jid
    jid=$(sbatch \
        -J "lt-${dataset:0:5}-${model:0:8}-s${seed}" \
        -o "$LOGS/lt_${tag}_%j.out" \
        -e "$LOGS/lt_${tag}_%j.err" \
        --export="DATASET=$dataset,MODELS=$model,SEED=$seed" \
        "$SCRIPT" | awk '{print $NF}')
    echo "  $jid  $model on $dataset seed=$seed  (~${est}h)"
}

echo "=== pathformer on electricity (3 × ~70h) ==="
for seed in 42 123 2026; do
    submit_seeded pathformer electricity $seed 70
done

echo ""
echo "=== spacetimeformer on traffic (3 × ~37h) ==="
for seed in 42 123 2026; do
    submit_seeded spacetimeformer traffic $seed 37
done

echo ""
echo "=== pathformer on traffic (3 × ~102h, borderline) ==="
for seed in 42 123 2026; do
    submit_seeded pathformer traffic $seed 102
done

echo ""
echo "Watch queue: squeue -u \$USER"
