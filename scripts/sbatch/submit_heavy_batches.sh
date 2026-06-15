#!/bin/bash
# Submit all electricity and traffic batch jobs.
#
# PREREQUISITE: cancel the monolithic lt-heavy job first:
#   scancel 149282
#
# Then run this script to submit all model-batched replacements:
#   bash scripts/sbatch/submit_heavy_batches.sh
#
# Each batch is sized to fit within the 4-day (96h) cluster time limit.
# Models within each batch are ordered fastest-first so the most results
# accumulate before any unexpected wall-clock hit.
#
# Timing estimates (A100, based on measured ETTh2 data scaled by channel count):
#
#   ELECTRICITY (321 channels, ×6 CI / ×15 CM vs ETTh2):
#     B1  ~84h  14 models (fast, channel-independent)
#     B2  ~79h  4 models  (medium)
#     B3  ~53h  2 models
#     B4  ~80h  2 models
#     B5  ~42h  crossformer
#     B6  ~66h  chronos2
#     B7  ~75h  lag_llama
#     B8  ~75h  spacetimeformer
#     ---  pathformer (~209h) is INFEASIBLE on electricity; excluded.
#
#   TRAFFIC (862 channels, ×8 CI / ×22 CM vs ETTh2):
#     B1  ~75h  12 models (fast)
#     B2  ~69h  3 models
#     B3  ~88h  3 models
#     B4  ~75h  2 models
#     B5  ~59h  contiformer
#     B6  ~59h  deformable_tst
#     B7  ~62h  crossformer
#     ---  chronos2/lag_llama/spacetimeformer/pathformer are INFEASIBLE on traffic; excluded.

set -e
REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
SCRIPT="$REPO/scripts/sbatch/run_lt_batch.sh"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

# ---------------------------------------------------------------------------
# ELECTRICITY
# ---------------------------------------------------------------------------

echo "=== Submitting electricity batches ==="

# B1: 15 fast models — channel-independent + small channel-mixing (~84h estimated)
# etsformer included here (similar speed to fedformer); absent from ETTh2 logs — may have crashed, needs re-run.
ELEC_B1="itransformer,timexer,transformer,patchtst,multipatchformer,pyraformer,nonstationary_transformer,cats,reformer,tft,informer,etsformer,fedformer,scaleformer,card"
JOB_ELEC_B1=$(sbatch \
    -J "lt-electricity-b1" \
    -o "$LOGS/lt_electricity_b1_%j.out" \
    -e "$LOGS/lt_electricity_b1_%j.err" \
    --export="DATASET=electricity,MODELS=$ELEC_B1" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B1 submitted: $JOB_ELEC_B1  (~84h, 15 models)"

# B2: medium models — basisformer, autoformer, triformer, airformer (~79h)
ELEC_B2="basisformer,autoformer,triformer,airformer"
JOB_ELEC_B2=$(sbatch \
    -J "lt-electricity-b2" \
    -o "$LOGS/lt_electricity_b2_%j.out" \
    -e "$LOGS/lt_electricity_b2_%j.err" \
    --export="DATASET=electricity,MODELS=$ELEC_B2" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B2 submitted: $JOB_ELEC_B2  (~79h, basisformer autoformer triformer airformer)"

# B3: quatformer + earthformer (~53h)
ELEC_B3="quatformer,earthformer"
JOB_ELEC_B3=$(sbatch \
    -J "lt-electricity-b3" \
    -o "$LOGS/lt_electricity_b3_%j.out" \
    -e "$LOGS/lt_electricity_b3_%j.err" \
    --export="DATASET=electricity,MODELS=$ELEC_B3" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B3 submitted: $JOB_ELEC_B3  (~53h, quatformer earthformer)"

# B4: contiformer + deformable_tst (~80h)
ELEC_B4="contiformer,deformable_tst"
JOB_ELEC_B4=$(sbatch \
    -J "lt-electricity-b4" \
    -o "$LOGS/lt_electricity_b4_%j.out" \
    -e "$LOGS/lt_electricity_b4_%j.err" \
    --export="DATASET=electricity,MODELS=$ELEC_B4" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B4 submitted: $JOB_ELEC_B4  (~80h, contiformer deformable_tst)"

# B5: crossformer alone (~42h)
JOB_ELEC_B5=$(sbatch \
    -J "lt-electricity-b5" \
    -o "$LOGS/lt_electricity_b5_%j.out" \
    -e "$LOGS/lt_electricity_b5_%j.err" \
    --export="DATASET=electricity,MODELS=crossformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B5 submitted: $JOB_ELEC_B5  (~42h, crossformer)"

# chronos2 and lag_llama (from-scratch) EXCLUDED — benchmark uses only pretrained
# versions (chronos_bolt, lag_llama_pretrained) via run_foundation.sh.

# B6: spacetimeformer alone (~75h)
JOB_ELEC_B6=$(sbatch \
    -J "lt-electricity-b6" \
    -o "$LOGS/lt_electricity_b6_%j.out" \
    -e "$LOGS/lt_electricity_b6_%j.err" \
    --export="DATASET=electricity,MODELS=spacetimeformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B6 submitted: $JOB_ELEC_B6  (~75h, spacetimeformer)"

echo "  NOTE: pathformer excluded (209h total; use submit_slow_seeded.sh for per-seed jobs)."

# ---------------------------------------------------------------------------
# TRAFFIC
# ---------------------------------------------------------------------------

echo ""
echo "=== Submitting traffic batches ==="

# B1: 13 fast models (~75h)
TRAFFIC_B1="itransformer,timexer,transformer,patchtst,multipatchformer,pyraformer,nonstationary_transformer,cats,reformer,tft,informer,etsformer,fedformer"
JOB_TRAFFIC_B1=$(sbatch \
    -J "lt-traffic-b1" \
    -o "$LOGS/lt_traffic_b1_%j.out" \
    -e "$LOGS/lt_traffic_b1_%j.err" \
    --export="DATASET=traffic,MODELS=$TRAFFIC_B1" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B1 submitted: $JOB_TRAFFIC_B1  (~75h, 13 models)"

# B2: scaleformer, card, triformer (~69h)
JOB_TRAFFIC_B2=$(sbatch \
    -J "lt-traffic-b2" \
    -o "$LOGS/lt_traffic_b2_%j.out" \
    -e "$LOGS/lt_traffic_b2_%j.err" \
    --export="DATASET=traffic,MODELS=scaleformer,card,triformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B2 submitted: $JOB_TRAFFIC_B2  (~69h, scaleformer card triformer)"

# B3: basisformer, autoformer, airformer (~88h)
JOB_TRAFFIC_B3=$(sbatch \
    -J "lt-traffic-b3" \
    -o "$LOGS/lt_traffic_b3_%j.out" \
    -e "$LOGS/lt_traffic_b3_%j.err" \
    --export="DATASET=traffic,MODELS=basisformer,autoformer,airformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B3 submitted: $JOB_TRAFFIC_B3  (~88h, basisformer autoformer airformer)"

# B4: quatformer + earthformer (~75h)
JOB_TRAFFIC_B4=$(sbatch \
    -J "lt-traffic-b4" \
    -o "$LOGS/lt_traffic_b4_%j.out" \
    -e "$LOGS/lt_traffic_b4_%j.err" \
    --export="DATASET=traffic,MODELS=quatformer,earthformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B4 submitted: $JOB_TRAFFIC_B4  (~75h, quatformer earthformer)"

# B5: contiformer (~59h)
JOB_TRAFFIC_B5=$(sbatch \
    -J "lt-traffic-b5" \
    -o "$LOGS/lt_traffic_b5_%j.out" \
    -e "$LOGS/lt_traffic_b5_%j.err" \
    --export="DATASET=traffic,MODELS=contiformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B5 submitted: $JOB_TRAFFIC_B5  (~59h, contiformer)"

# B6: deformable_tst (~59h)
JOB_TRAFFIC_B6=$(sbatch \
    -J "lt-traffic-b6" \
    -o "$LOGS/lt_traffic_b6_%j.out" \
    -e "$LOGS/lt_traffic_b6_%j.err" \
    --export="DATASET=traffic,MODELS=deformable_tst" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B6 submitted: $JOB_TRAFFIC_B6  (~59h, deformable_tst)"

# B7: crossformer (~62h)
JOB_TRAFFIC_B7=$(sbatch \
    -J "lt-traffic-b7" \
    -o "$LOGS/lt_traffic_b7_%j.out" \
    -e "$LOGS/lt_traffic_b7_%j.err" \
    --export="DATASET=traffic,MODELS=crossformer" \
    "$SCRIPT" | awk '{print $NF}')
echo "  B7 submitted: $JOB_TRAFFIC_B7  (~62h, crossformer)"

echo "  NOTE: spacetimeformer and pathformer excluded from traffic main batches."
echo "        Use submit_slow_seeded.sh for per-seed jobs of those two models."
echo "        chronos2 and lag_llama (from-scratch) not in scope — use pretrained versions."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=== Job IDs submitted ==="
echo "  Electricity: $JOB_ELEC_B1 $JOB_ELEC_B2 $JOB_ELEC_B3 $JOB_ELEC_B4 $JOB_ELEC_B5 $JOB_ELEC_B6"
echo "  Traffic:     $JOB_TRAFFIC_B1 $JOB_TRAFFIC_B2 $JOB_TRAFFIC_B3 $JOB_TRAFFIC_B4 $JOB_TRAFFIC_B5 $JOB_TRAFFIC_B6 $JOB_TRAFFIC_B7"
echo ""
echo "Watch queue: squeue -u \$USER"
