#!/bin/bash
# NOTE: spacetimeformer and pathformer are now handled inside submit_heavy_batches.sh
# via run_heavy_batch.sh arrays. This file is kept only as a reference for
# submitting individual per-seed jobs manually if needed (e.g. to rerun a failed seed).
#
# chronos2 and lag_llama (from-scratch) are NOT included — use chronos_bolt and
# lag_llama_pretrained (pretrained) instead.
#
# Usage (manual resubmission of a single failed seed):
#   sbatch -J "heavy-pathformer-s42" \
#          -o logs/heavy_pathformer_s42_%A_%a.out \
#          -e logs/heavy_pathformer_s42_%A_%a.err \
#          --export=MODELS=pathformer,SEED=42 \
#          scripts/sbatch/run_heavy_batch.sh

echo "See submit_heavy_batches.sh for the canonical submission of slow models."
echo "This file is a reference only — run submit_heavy_batches.sh instead."
