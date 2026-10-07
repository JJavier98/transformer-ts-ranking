#!/bin/bash
# Gap-fill for the CORE ranking scope (deadline-driven scope reduction,
# 2026-10-05).
#
# The main ranking (Friedman / Nemenyi / CD diagram) is restricted to the 7
# datasets that every eligible model can complete within the cluster wall-time:
#   ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, illness, weather.
# electricity and traffic are reported as a partial supplementary table only.
# contiformer is reported as "not measured" (OOM at every batch size, even
# batch=1 on A100) and excluded from the statistical tests.
#
# This script runs exactly the 47 (model, dataset, horizon, seed) cells still
# missing in that scope, plus the 3 missing M4 cells (triformer, Monthly).
# The cell list was computed from the persisted parquet shards on 2026-10-05.
# Every cell keeps the configuration used for the rest of its dataset
# (paper_ready, fp32, batch_size=16) so the new rows are mixable with the old.
#
# Task table (cost-ascending: fast models first, autoregressive lag_llama last):
#   0-2   exchange_rate h720, seeds 42/123/2026      (fast + slow models)
#   3-5   weather h720,       seeds 42/123/2026      (fast models)
#   6     M4 Monthly, triformer, seeds 42/123/2026
#   7-10  weather, pathformer (4 cells: OOM / ValueError on V100 at bs=16)
#   11-13 weather h720, lag_llama_pretrained, seeds 42/123/2026
#   14-17 weather, pathformer, REPLACEMENT seed 7 (see below)
#   18-29 weather, airformer, all seeds/horizons, with the log-variance clamp
#         (the 12 original runs ended with NaN metrics)
#   30-32 airformer with the clamp on ETTh1 h96, exchange_rate h96 and
#         illness h24 (seed 42): check that the clamp does not change the
#         results of cells that already worked
#
# Replacement-seed rule (fixed 2026-10-06, before any replacement result was
# seen).  pathformer diverged numerically on 4 weather cells (h96/s42,
# h192/s42, h336/s2026, h720/s2026): its noisy gating receives NaN logits
# and torch.distributions.Normal raises a ValueError.  This is reproducible
# for a given seed and is not a memory problem.  A diverged run is replaced
# by seed 7; if seed 7 also diverges, by seed 99; if that also diverges, the
# cell keeps the remaining 2 seeds.  The paper reports every failed run and
# its replacement.  CORRECTION (2026-10-07): airformer also diverged on weather,
# silently (NaN metrics, no exception, so it was counted as a valid run); see
# configs/benchmark/report_scope.yaml.
#
# pathformer tasks (7-10) must run on the A100 node to fit at the canonical
# batch_size=16, so submit them separately pinned to the dgx2 partition:
#   sbatch --array=0-6,11-13 scripts/sbatch/run_fill_core.sh
#   sbatch --array=7-10 --partition=dgx2 scripts/sbatch/run_fill_core.sh
#   sbatch --array=14-17 --partition=dgx2 scripts/sbatch/run_fill_core.sh
# Cells that still fail are marked N/A (documented) in the ranking.
#
# Results layout:
#   long-term: results/raw/long_term/<DATASET>_s<SEED>_fill<T>_h<H>  (T = task index, so tasks sharing a cell key never share a parquet)/results_raw.parquet
#   M4:        results/raw/m4/Monthly_fill/results_raw.parquet
# merge_results.py keeps the latest run_timestamp per run key, so these rows
# supersede the earlier error rows for the same cells.

#SBATCH --job-name fill-core
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-32
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/fill_core_%A_%a.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/fill_core_%A_%a.err

# One entry per array index: "task|dataset|seed|horizon|models[|batch_size]".
# batch_size defaults to the canonical 16.  lag_llama_pretrained uses 256: it
# is zero-shot (no-op fit), the test loader has drop_last=False and each series
# is normalised independently, so batch size only changes throughput.
# Validated 2026-10-05 (job 177685, scripts/sbatch/validate_lag_llama_bs.sh):
# ETTh1 h96 MAE 14.578375 at bs=256 vs 14.578377 at bs=16, 6 min vs ~1.5 h.
TASKS=(
    "long_term|exchange_rate|42|720|tft,timexer,transformer,triformer"
    "long_term|exchange_rate|123|720|tft,timexer,transformer,triformer"
    "long_term|exchange_rate|2026|720|spacetimeformer,tft,timexer,transformer,triformer"
    "long_term|weather|42|720|multipatchformer,nonstationary_transformer,patchtst,pyraformer,reformer,scaleformer,tft,timexer,transformer"
    "long_term|weather|123|720|multipatchformer,nonstationary_transformer,patchtst,pyraformer,reformer,scaleformer,tft,timexer,transformer"
    "long_term|weather|2026|720|multipatchformer,nonstationary_transformer,patchtst,pyraformer,reformer,scaleformer,tft,timexer,transformer"
    "m4|Monthly|42,123,2026|-|triformer"
    "long_term|weather|42|96|pathformer"
    "long_term|weather|42|192|pathformer"
    "long_term|weather|2026|336|pathformer"
    "long_term|weather|2026|720|pathformer"
    "long_term|weather|42|720|lag_llama_pretrained|256"
    "long_term|weather|123|720|lag_llama_pretrained|256"
    "long_term|weather|2026|720|lag_llama_pretrained|256"
    "long_term|weather|7|96|pathformer"
    "long_term|weather|7|192|pathformer"
    "long_term|weather|7|336|pathformer"
    "long_term|weather|7|720|pathformer"
    "long_term|weather|42|96|airformer"
    "long_term|weather|42|192|airformer"
    "long_term|weather|42|336|airformer"
    "long_term|weather|42|720|airformer"
    "long_term|weather|123|96|airformer"
    "long_term|weather|123|192|airformer"
    "long_term|weather|123|336|airformer"
    "long_term|weather|123|720|airformer"
    "long_term|weather|2026|96|airformer"
    "long_term|weather|2026|192|airformer"
    "long_term|weather|2026|336|airformer"
    "long_term|weather|2026|720|airformer"
    "long_term|ETTh1|42|96|airformer"
    "long_term|exchange_rate|42|96|airformer"
    "long_term|illness|42|24|airformer"
)

T=$SLURM_ARRAY_TASK_ID
IFS='|' read -r TASK DATASET SEEDS HORIZON MODELS BS <<< "${TASKS[$T]}"
BS=${BS:-16}

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID/$T] fill-core: $TASK $DATASET seeds=$SEEDS h=$HORIZON models=$MODELS bs=$BS"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

if [ "$TASK" = "m4" ]; then
    # M4 horizons are fixed per frequency by the experiment config.
    "$PYTHON" "$REPO/scripts/launch_benchmark.py" \
        training=paper_ready \
        "training.seeds=[$SEEDS]" \
        training.batch_size=16 \
        experiment=m4 \
        "experiment.datasets=[$DATASET]" \
        "experiment.models=[$MODELS]" \
        wandb=disabled \
        "+results_dir=results/raw/m4/${DATASET}_fill" \
        "hydra.run.dir=outputs/m4/${DATASET}_fill"
else
    "$PYTHON" "$REPO/scripts/launch_benchmark.py" \
        training=paper_ready \
        "training.seeds=[$SEEDS]" \
        "training.batch_size=$BS" \
        experiment=long_term \
        "experiment.datasets=[$DATASET]" \
        "experiment.models=[$MODELS]" \
        "experiment.horizons=[$HORIZON]" \
        wandb=disabled \
        "+results_dir=results/raw/long_term/${DATASET}_s${SEEDS}_fill${T}_h${HORIZON}" \
        "hydra.run.dir=outputs/lt/${DATASET}_s${SEEDS}_fill${T}_h${HORIZON}"
fi
STATUS=$?

# Propagate the Python exit code so a crash surfaces as FAILED in sacct.
echo "[$SLURM_JOB_ID/$T] Finished: $TASK $DATASET seeds=$SEEDS h=$HORIZON (exit=$STATUS)"
exit $STATUS
