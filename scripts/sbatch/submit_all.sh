#!/bin/bash
# Submit all benchmark job arrays to SLURM, respecting the 4-concurrent-process limit.
#
# Strategy:
#   1. Long-term (7 lighter datasets) starts immediately on dgx, max 4 concurrent.
#   2. Long-term heavy (electricity, traffic) starts immediately on dgx2, max 2 concurrent.
#   3. M4 (6 frequencies) starts AFTER all long-term jobs finish (--dependency),
#      max 4 concurrent on dgx.
#   4. Merge job runs after both M4 arrays finish.
#
# Maximum concurrent at any time = 4 (long-term) + 2 (heavy) = 6 during phase 1,
# then 4 (M4) during phase 2.  If the cluster's 4-process limit applies across ALL
# partitions, reduce --array to %2 in the long-term scripts and %1 in the heavy script.
#
# Usage:
#   bash scripts/sbatch/submit_all.sh

set -e

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
cd "$REPO"
mkdir -p "$REPO/logs"

echo "=== Submitting long-term benchmark (dgx, 7 datasets) ==="
LT_JOB=$(sbatch scripts/sbatch/run_long_term.sh | awk '{print $4}')
echo "  Submitted array job: $LT_JOB"

echo "=== Submitting heavy long-term benchmark (dgx2, electricity + traffic) ==="
LT_HEAVY_JOB=$(sbatch scripts/sbatch/run_long_term_heavy.sh | awk '{print $4}')
echo "  Submitted array job: $LT_HEAVY_JOB"

echo "=== Submitting M4 benchmark (dgx, 6 frequencies) — starts after long-term ==="
M4_JOB=$(sbatch --dependency=afterok:${LT_JOB}:${LT_HEAVY_JOB} scripts/sbatch/run_m4.sh | awk '{print $4}')
echo "  Submitted array job: $M4_JOB (depends on $LT_JOB, $LT_HEAVY_JOB)"

echo ""
echo "=== All jobs submitted ==="
echo "  Long-term (7 light):  $LT_JOB"
echo "  Long-term (2 heavy):  $LT_HEAVY_JOB"
echo "  M4 (6 frequencies):   $M4_JOB"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "View logs:     tail -f $REPO/logs/long_term_${LT_JOB}_*.out"
echo ""
echo "After all jobs finish, merge results with:"
echo "  $REPO/.venv/bin/python $REPO/scripts/sbatch/merge_results.py"
