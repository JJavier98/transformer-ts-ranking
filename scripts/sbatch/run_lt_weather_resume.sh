#!/bin/bash
# Long-term benchmark — weather seed=42 RESUME after timeout.
#
# Job 151198_0 ran for 3d 6h and timed out before writing the final
# parquet.  The runner's incremental checkpoint (added 2026-06-24) now
# writes the parquet after every combo and skips already-done combos
# on re-entry.
#
# This job simply resubmits the same run.  The runner will detect the
# existing results_raw.parquet (if any partial results were written by a
# previous resumed run) or start fresh if the directory is empty.
# No manual model or horizon filtering is needed.
#
# Usage:
#   sbatch scripts/sbatch/run_lt_weather_resume.sh

#SBATCH --job-name lt-weather-s42
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_weather_resume_%j.out
#SBATCH -e /mnt/homeGPU/JJavierAR/transformer-ts-ranking/logs/lt_weather_resume_%j.err

REPO=/mnt/homeGPU/JJavierAR/transformer-ts-ranking
PYTHON="$REPO/.venv/bin/python"

export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

echo "[$SLURM_JOB_ID] lt-weather seed=42 RESUME — incremental checkpoint active"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $($PYTHON --version 2>&1)"

cd "$REPO"
mkdir -p "$REPO/logs"

"$PYTHON" "$REPO/scripts/launch_benchmark.py" \
    training=paper_ready \
    "training.seeds=[42]" \
    training.batch_size=16 \
    experiment=long_term \
    "experiment.datasets=[weather]" \
    wandb=disabled \
    "+results_dir=results/raw/long_term/weather_s42" \
    "hydra.run.dir=outputs/lt/weather_s42_resume"

echo "[$SLURM_JOB_ID] Finished: weather seed=42"
