#!/bin/bash
# Ordered resume submission — "slowest last" so the fast combinations return
# first and we don't wait on the long ones.
#
# The cluster allows only 4 concurrent jobs (AssocMaxJobsLimit). We submit in
# tiers with increasing --nice (higher nice = lower priority = scheduled later):
# SLURM always prefers a lower-nice pending task for the 4 slots, so the faster
# tiers drain first and the slow tiers fill in only afterwards (backfill may
# still use genuinely idle slots without delaying the faster tasks). Combined
# with submit order (lower job ids first), this puts the presumed-longest work
# dead last.
#
# WITHIN each array the indices are already ordered cheap-first (small-channel
# datasets before electricity/traffic); when the lag_llama array is created it
# MUST also be cost-ascending so traffic/h720 (possibly N/A) come last.
#
# Run AFTER the current jobs finish, from anywhere:
#   bash scripts/sbatch/submit_resume.sh
#
# Tiers (nice → later):
#   1 (nice 0)      22 fast models — gap-fill electricity/traffic
#   2 (nice 100)    4 slow-but-memory-OK models (triformer/quatformer/spacetimeformer/deformable_tst)
#   3 (nice 500)    contiformer (ctx=48, bs=1) + pathformer (bs=2 on heavy) — unitary, slow
#   4 (nice 10000)  lag_llama (autoregressive) — SLOWEST, runs dead last
#
# See the 'benchmark-resume-plan' memory for the full context and the lag_llama
# TODO (its isolated unitary script does not exist yet).

set -u
cd "$(dirname "$0")/../.." || exit 1
S=scripts/sbatch

submit() {  # submit <nice> <script>
    local nice=$1 script=$2
    if [[ -f "$script" ]]; then
        echo "  sbatch --nice=$nice $script"
        sbatch --nice="$nice" "$script"
    else
        echo "  SKIP (missing — create it first): $script"
    fi
}

echo "== Tier 0 (nice 0): toto2 — zero-shot, seconds per cell, cheapest first =="
submit 0 "$S/run_lt_toto2.sh"

echo "== Tier 1 (nice 0): fast 22 models — electricity/traffic gap-fill =="
submit 0 "$S/run_lt_heavy.sh"

echo "== Tier 2 (nice 100): 4 slow-but-OK models =="
submit 100 "$S/run_lt_slow.sh"

echo "== Tier 3 (nice 500): contiformer + pathformer (unitary, slow) =="
submit 500 "$S/run_lt_contiformer.sh"
submit 500 "$S/run_lt_pathformer.sh"

echo "== Tier 4 (nice 10000): lag_llama — SLOWEST, dead last =="
submit 10000 "$S/run_lt_lag_llama.sh"

echo ""
echo "Submitted. 'squeue -u \$USER --sort=+p' shows priority order (nice raises the number)."
