#!/bin/bash
# Submit all electricity + traffic batch jobs as SLURM arrays.
# Each array has task 0 = electricity, task 1 = traffic; both run in parallel.
#
# Model batches are designed to fit within the 4-day (96h) cluster limit for
# the heavier dataset (traffic, 862 channels). Estimates use measured ETTh2
# timings scaled by channel count (×6 CI / ×15 CM for electricity; ×8/×22 traffic).
#
# Standard batches (7 array jobs, 2 tasks each = 14 GPU-jobs total):
#
#   B1  ~75h/traffic  12 models — fast, channel-independent
#   B2  ~86h/traffic   4 models — medium (scaleformer, card, etsformer, triformer)
#   B3  ~88h/traffic   3 models — basisformer, autoformer, airformer
#   B4  ~75h/traffic   2 models — quatformer, earthformer
#   B5  ~59h/traffic   1 model  — contiformer
#   B6  ~59h/traffic   1 model  — deformable_tst
#   B7  ~62h/traffic   1 model  — crossformer
#
# Per-seed batches (3 array jobs × 2 seeds/model × 2 models = 12 GPU-jobs):
#   spacetimeformer — ~75h/seed electricity, ~37h/seed traffic   ✅
#   pathformer      — ~70h/seed electricity, ~102h/seed traffic  ⚠ borderline on traffic
#
# chronos2 and lag_llama (from-scratch) are NOT included — the benchmark uses
# only their pretrained versions (chronos_bolt, lag_llama_pretrained).
#
# Usage:
#   bash scripts/sbatch/submit_heavy_batches.sh

set -e
REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
SCRIPT="$REPO/scripts/sbatch/run_heavy_batch.sh"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

submit_array() {
    local tag="$1" models="$2" seed="${3:-}" est_elec="$4" est_traffic="$5"
    local seed_export=""
    local seed_label="all seeds"
    if [[ -n "$seed" ]]; then
        seed_export=",SEED=$seed"
        seed_label="seed=$seed"
    fi
    local jid
    jid=$(sbatch \
        -J "heavy-${tag}" \
        -o "$LOGS/heavy_${tag}_%A_%a.out" \
        -e "$LOGS/heavy_${tag}_%A_%a.err" \
        --export="MODELS=$models${seed_export}" \
        "$SCRIPT" | awk '{print $NF}')
    echo "  $jid  [${tag}] $seed_label  elec~${est_elec}h  traffic~${est_traffic}h"
}

# ---------------------------------------------------------------------------
# Standard batches — all 3 seeds, array[electricity, traffic]
# ---------------------------------------------------------------------------
echo "=== Standard batches (arrays of 2) ==="

# B1: 12 fast models (~50h electricity / ~75h traffic)
B1="itransformer,timexer,transformer,patchtst,multipatchformer,pyraformer,nonstationary_transformer,cats,reformer,tft,informer,fedformer"
submit_array "b1" "$B1" "" 50 75

# B2: scaleformer, card, etsformer, triformer (~57h electricity / ~86h traffic)
submit_array "b2" "scaleformer,card,etsformer,triformer" "" 57 86

# B3: basisformer, autoformer, airformer (~58h electricity / ~88h traffic)
submit_array "b3" "basisformer,autoformer,airformer" "" 58 88

# B4: quatformer, earthformer (~53h electricity / ~75h traffic)
submit_array "b4" "quatformer,earthformer" "" 53 75

# B5: contiformer (~40h electricity / ~59h traffic)
submit_array "b5" "contiformer" "" 40 59

# B6: deformable_tst (~40h electricity / ~59h traffic)
submit_array "b6" "deformable_tst" "" 40 59

# B7: crossformer (~42h electricity / ~62h traffic)
submit_array "b7" "crossformer" "" 42 62

# ---------------------------------------------------------------------------
# Per-seed batches — spacetimeformer and pathformer exceed 96h with all seeds
# ---------------------------------------------------------------------------
echo ""
echo "=== Per-seed batches (arrays of 2, 3 submissions per model) ==="

for seed in 42 123 2026; do
    # spacetimeformer: ~75h/seed electricity, ~37h/seed traffic
    submit_array "spacetimeformer-s${seed}" "spacetimeformer" "$seed" 75 37
done

for seed in 42 123 2026; do
    # pathformer: ~70h/seed electricity, ~102h/seed traffic (borderline)
    submit_array "pathformer-s${seed}" "pathformer" "$seed" 70 102
done

echo ""
echo "Watch queue: squeue -u \$USER"
