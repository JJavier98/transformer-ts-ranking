"""Merge per-job result parquets and JSONL logs into the canonical output files.

Each SBATCH job writes to its own subdirectory to avoid write contention:
  results/raw/long_term/<DATASET>/results_raw.parquet
  results/raw/long_term/<DATASET>/epoch_logs.jsonl
  results/raw/m4/<FREQUENCY>/results_raw.parquet
  results/raw/m4/<FREQUENCY>/epoch_logs.jsonl

This script merges all shards into:
  results/raw/results_raw.parquet   (all rows from all jobs)
  results/raw/epoch_logs.jsonl      (all epoch log lines)

Usage::

    conda run -n torch_env python scripts/sbatch/merge_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "results" / "raw"

sys.path.insert(0, str(REPO_ROOT / "src"))


def merge_parquets(shard_dirs: list[Path], output_path: Path) -> int:
    """Concatenate all shard parquets into one file.

    Args:
        shard_dirs: Directories that may contain ``results_raw.parquet``.
        output_path: Destination parquet file.

    Returns:
        Total number of rows written.
    """
    frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        p = shard_dir / "results_raw.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            print(f"  [skip] no parquet in {shard_dir.name}")

    if not frames:
        print("No parquet shards found.")
        return 0

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"  Written {len(combined)} rows to {output_path.relative_to(REPO_ROOT)}")
    return len(combined)


def merge_jsonl(shard_dirs: list[Path], output_path: Path) -> int:
    """Concatenate all shard JSONL files into one file.

    Args:
        shard_dirs: Directories that may contain ``epoch_logs.jsonl``.
        output_path: Destination JSONL file.

    Returns:
        Total number of lines written.
    """
    total_lines = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh_out:
        for shard_dir in shard_dirs:
            p = shard_dir / "epoch_logs.jsonl"
            if not p.exists():
                continue
            with p.open() as fh_in:
                for line in fh_in:
                    fh_out.write(line)
                    total_lines += 1

    print(f"  Written {total_lines} epoch log lines to {output_path.relative_to(REPO_ROOT)}")
    return total_lines


def main() -> None:
    """Merge all shards from long_term and m4 subdirectories."""
    print("=== Merging long-term results ===")
    lt_dir = RAW_DIR / "long_term"
    lt_shards = sorted(lt_dir.iterdir()) if lt_dir.exists() else []
    merge_parquets(lt_shards, lt_dir / "results_raw.parquet")
    merge_jsonl(lt_shards, lt_dir / "epoch_logs.jsonl")

    print("\n=== Merging M4 results ===")
    m4_dir = RAW_DIR / "m4"
    m4_shards = sorted(m4_dir.iterdir()) if m4_dir.exists() else []
    merge_parquets(m4_shards, m4_dir / "results_raw.parquet")
    merge_jsonl(m4_shards, m4_dir / "epoch_logs.jsonl")

    print("\n=== Merging all tracks into canonical location ===")
    all_shards = lt_shards + m4_shards
    merge_parquets(all_shards, RAW_DIR / "results_raw.parquet")
    merge_jsonl(all_shards, RAW_DIR / "epoch_logs.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
