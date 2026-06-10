#!/usr/bin/env bash
# Smoke driver for transformer-ts-ranking.
# Run from the repo root.  Requires the torch_env conda environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="$REPO_ROOT/.claude/skills/run-transformer-ts-ranking/runner.py"

run_cmd() {
    local label="$1"; shift
    echo "==> $label"
    conda run -n torch_env python "$RUNNER" "$@"
    echo ""
}

cd "$REPO_ROOT"

run_cmd "audit-models"               audit-models
run_cmd "materialize-manifests"      materialize-manifests
run_cmd "smoke-long-term (ETTh1)"    smoke-long-term --dataset ETTh1
run_cmd "smoke-m4 (Hourly)"          smoke-m4 --frequency Hourly

# Subset --models for speed; omit to probe all 29 (much slower)
run_cmd "probe-compatibility"        probe-compatibility --models patchtst,autoformer
run_cmd "validate-canonical-forward" validate-canonical-forward --models patchtst,autoformer

echo "All smoke checks passed."
