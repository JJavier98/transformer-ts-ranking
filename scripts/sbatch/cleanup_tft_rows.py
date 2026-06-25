#!/usr/bin/env python3
"""Purge corrupted TFT rows written before the _SEQ2SEQ_MODELS fix.

Jobs 151198_1 (weather_s123), 151198_2 (weather_s2026), 151198_3 (ETTm1_s42),
and 151198_5 (ETTm1_s2026) ran with the old code where 'tft' was missing from
_SEQ2SEQ_MODELS.  With label_len=0, Python's x[:, -0:, :] returns the full
encoder sequence instead of the last 48 steps — silently corrupting TFT outputs.

Run this script AFTER all four jobs above have finished (check with `squeue -u $USER`).
Then submit scripts/sbatch/run_tft_rerun.sh to recompute the correct TFT results.

Usage:
    .venv/bin/python scripts/sbatch/cleanup_tft_rows.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# Parquet shards written by running jobs (old code, wrong TFT rows).
# exchange_rate and illness were already cleaned immediately after the bug was
# discovered.  This script covers the lt-medium shards produced by old-code jobs.
SHARDS_TO_CLEAN = [
    REPO / "results/raw/long_term/weather_s123/results_raw.parquet",
    REPO / "results/raw/long_term/weather_s2026/results_raw.parquet",
    REPO / "results/raw/long_term/ETTm1_s42/results_raw.parquet",
    REPO / "results/raw/long_term/ETTm1_s2026/results_raw.parquet",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without writing anything.",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not available. Run with .venv/bin/python", file=sys.stderr)
        sys.exit(1)

    total_purged = 0
    for shard in SHARDS_TO_CLEAN:
        if not shard.exists():
            print(f"[SKIP — not found] {shard.relative_to(REPO)}")
            continue

        df = pd.read_parquet(shard)
        tft_mask = df["model_name"] == "tft"
        n_tft = tft_mask.sum()

        if n_tft == 0:
            print(f"[OK — no TFT rows] {shard.relative_to(REPO)}")
            continue

        tft_rows = df[tft_mask][["model_name", "dataset_name", "horizon", "seed", "error"]]
        print(f"\n[PURGE] {shard.relative_to(REPO)}: {n_tft} TFT rows to remove")
        print(tft_rows.to_string(index=False))

        if not args.dry_run:
            df_clean = df[~tft_mask].reset_index(drop=True)
            df_clean.to_parquet(shard, index=False)
            print(f"  → written ({len(df)} → {len(df_clean)} rows)")
            total_purged += n_tft
        else:
            print("  → (dry-run, not written)")
            total_purged += n_tft

    if args.dry_run:
        print(f"\nDRY RUN: would purge {total_purged} TFT rows total.")
        print("Rerun without --dry-run to apply, then submit run_tft_rerun.sh")
    else:
        print(f"\nPurged {total_purged} TFT rows total.")
        if total_purged > 0:
            print("Next: sbatch scripts/sbatch/run_tft_rerun.sh")


if __name__ == "__main__":
    main()
