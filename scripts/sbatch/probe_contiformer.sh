#!/bin/bash
# ContiFormer / ContiFormer-fast: library checks + cost probe (GPU node only).
#
# All stages run on the GPU node (nothing runs on the head node):
#   1. Library unit tests: contiformer, contiformer_fast, and the
#      repository-wide registry/API tests that cover every registered model.
#   2. capabilities.yaml is regenerated and must be unchanged (the
#      contiformer_fast entry was added by hand in the generator's format).
#   3. Numerical check (scripts/validate_contiformer_ckpt.py): new ContiFormer
#      with/without checkpointing vs the library before the changes
#      (OLD_COMMIT, node-local worktree), and ContiFormer-fast vs ContiFormer
#      with identical weights.
#   4. Cost probe (scripts/probe_contiformer.py): memory and time per training
#      step, epoch-time and full-core-grid extrapolation, written to
#      artifacts/review/contiformer_probe.json.
# It does not produce benchmark rows.
#
# Runs on any GPU node (A100 or V100; the report records which):
#   sbatch scripts/sbatch/probe_contiformer.sh

#SBATCH --job-name conti-probe
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/conti_probe_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/conti_probe_%j.err

# Library commit before any ContiFormer change.
OLD_COMMIT=4ed2ca2

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
LIB="$REPO/s-transformers-lib"
PYTHON="$REPO/.venv/bin/python"
WORK="${TMPDIR:-/tmp}/conti_check_$SLURM_JOB_ID"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID] contiformer checks + probe"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Library: $(git -C "$LIB" rev-parse --short HEAD) ($(git -C "$LIB" rev-parse --abbrev-ref HEAD)), dirty files: $(git -C "$LIB" status --short | wc -l)"
cd "$REPO"
mkdir -p "$WORK"

echo "=== 1. library unit tests ==="
(cd "$LIB" && "$PYTHON" -m pytest -q --tb=short \
    tests/test_contiformer.py tests/test_contiformer_fast.py \
    tests/test_api_contract_all_models.py \
    tests/test_public_api.py \
    "tests/test_predict_api.py::test_predict_all_canonical_models[contiformer]" \
    "tests/test_predict_api.py::test_predict_all_canonical_models[contiformer_fast]")
TESTS=$?
# The repository-wide minimum-standard test already fails upstream for
# chronos_bolt and lag_llama_pretrained (no dedicated test files).  Run it
# for information only and check that contiformer_fast is not in the report.
(cd "$LIB" && "$PYTHON" -m pytest -q --tb=line tests/test_minimum_testing_standard.py 2>&1 | tail -5)

echo "=== 2. capabilities.yaml regeneration ==="
cp "$LIB/s_transformers_lib/capabilities.yaml" "$WORK/capabilities.before.yaml"
(cd "$LIB" && "$PYTHON" scripts/gen_capabilities.py)
diff "$WORK/capabilities.before.yaml" "$LIB/s_transformers_lib/capabilities.yaml"
CAPS=$?
cp "$WORK/capabilities.before.yaml" "$LIB/s_transformers_lib/capabilities.yaml"

echo "=== 3. numerical check: old ($OLD_COMMIT) vs nockpt vs ckpt vs fast ==="
git -C "$LIB" worktree add -q --detach "$WORK/oldlib" "$OLD_COMMIT"
V=scripts/validate_contiformer_ckpt.py
"$PYTHON" $V run --lib "$WORK/oldlib" --mode old    --out "$WORK/old.pt" \
 && "$PYTHON" $V run --lib "$LIB"     --mode nockpt --out "$WORK/nockpt.pt" \
 && "$PYTHON" $V run --lib "$LIB"     --mode ckpt   --out "$WORK/ckpt.pt" \
 && "$PYTHON" $V run --lib "$LIB"     --mode fast   --out "$WORK/fast.pt" \
 && "$PYTHON" $V run --lib "$LIB"     --mode fast1  --out "$WORK/fast1.pt" \
 && "$PYTHON" $V run --lib "$LIB"     --mode fast1g --out "$WORK/fast1g.pt" \
 && "$PYTHON" $V compare --dir "$WORK"
CHECK=$?
git -C "$LIB" worktree remove --force "$WORK/oldlib"

echo "=== 4. cost probe ==="
"$PYTHON" scripts/probe_contiformer.py --out "${PROBE_OUT:-artifacts/review/contiformer_probe.json}"
PROBE=$?

rm -rf "$WORK"
echo "[$SLURM_JOB_ID] Finished: tests=$TESTS caps=$CAPS check=$CHECK probe=$PROBE"
[ $TESTS -eq 0 ] && [ $CAPS -eq 0 ] && [ $CHECK -eq 0 ] && [ $PROBE -eq 0 ]
