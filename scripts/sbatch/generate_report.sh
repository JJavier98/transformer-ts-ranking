#!/bin/bash
# Paper report: tables, statistics, figures and manifest from the persisted
# result shards (no model is run).  Executed on a compute node, never on the
# head node.  Optionally runs the reporting unit tests first.
#
# Usage:
#   sbatch scripts/sbatch/generate_report.sh            # report only
#   sbatch --export=ALL,RUN_TESTS=1 scripts/sbatch/generate_report.sh

#SBATCH --job-name report
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/report_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/report_%j.err

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"
cd "$REPO"
export MPLBACKEND=Agg

if [ "${RUN_TESTS:-0}" = "1" ]; then
    echo "=== reporting tests ==="
    "$PYTHON" -c "import sys, pytest; sys.path.insert(0, 'src'); sys.exit(pytest.main(['-q', '--tb=short', 'tests/test_reporting.py']))" || exit 1
fi

echo "=== generate-report ==="
"$PYTHON" .claude/skills/run-transformer-ts-ranking/runner.py generate-report
