#!/bin/bash
# Submit the full benchmark experiment after smoke tests pass.
#
# Run smoke tests first and verify 0 failures before calling this script:
#   sbatch scripts/sbatch/smoke_long_term.sh
#   sbatch scripts/sbatch/smoke_m4.sh
#
# Then inspect smoke results:
#   .venv/bin/python -c "
#   import pandas as pd
#   for p in ['results/smoke/long_term/ETTh1/results_raw.parquet',
#             'results/smoke/m4/Monthly/results_raw.parquet']:
#       df = pd.read_parquet(p)
#       ok = df['error'].isna().sum()
#       fail = df['error'].notna().sum()
#       print(f'{p}: OK={ok} FAIL={fail}')
#       if fail > 0:
#           print(df[df['error'].notna()][['model_name','horizon','error']].to_string())
#   "
#
# If all models pass, submit the full experiment:
#   bash scripts/sbatch/submit_experiment.sh
#
# Job structure:
#   lt-small  (4 tasks)  : illness, exchange_rate, ETTh1, ETTh2  — all seeds per task
#   lt-medium (9 tasks)  : weather, ETTm1, ETTm2                  — per-seed tasks
#   lt-heavy  (6 tasks)  : electricity, traffic                    — per-seed tasks
#   m4-full   (6 tasks)  : Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
#
# Total: 25 SLURM tasks, max 4 concurrent (AssocMaxJobsLimit).
#
# After all jobs complete, merge shards:
#   .venv/bin/python scripts/sbatch/merge_results.py

set -e

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking

echo "Submitting full benchmark experiment..."
echo ""

JOB1=$(sbatch --parsable "$REPO/scripts/sbatch/run_lt_small.sh")
echo "  lt-small  submitted: job $JOB1"

JOB2=$(sbatch --parsable "$REPO/scripts/sbatch/run_lt_medium.sh")
echo "  lt-medium submitted: job $JOB2"

JOB3=$(sbatch --parsable "$REPO/scripts/sbatch/run_lt_heavy.sh")
echo "  lt-heavy  submitted: job $JOB3"

JOB4=$(sbatch --parsable "$REPO/scripts/sbatch/run_m4_full.sh")
echo "  m4-full   submitted: job $JOB4"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER --format='%.10i %.12j %.8T %.12M %R'"
echo ""
echo "After all complete, merge shards:"
echo "  .venv/bin/python scripts/sbatch/merge_results.py"
