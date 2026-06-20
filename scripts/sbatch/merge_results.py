"""Merge per-job result parquets and JSONL logs into the canonical output files.

Each SBATCH job writes to its own subdirectory to avoid write contention:

  Long-term (small datasets — all seeds in one task):
    results/raw/long_term/<DATASET>/results_raw.parquet

  Long-term (medium/heavy — split by seed):
    results/raw/long_term/<DATASET>_s<SEED>/results_raw.parquet

  M4:
    results/raw/m4/<FREQ>/results_raw.parquet

This script merges all shards into:
  results/raw/long_term/results_raw.parquet  (all long-term rows)
  results/raw/m4/results_raw.parquet         (all M4 rows)
  results/raw/results_raw.parquet            (everything — canonical file)

Duplicate rows (same model/dataset/horizon/seed/task) are resolved by
keeping the most-recent run_timestamp so that targeted re-runs cleanly
replace stale rows.

Usage::

    .venv/bin/python scripts/sbatch/merge_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "results" / "raw"

sys.path.insert(0, str(REPO_ROOT / "src"))

_RUN_KEY = ["model_name", "dataset_name", "horizon", "seed", "task"]


def _shard_dirs(parent: Path) -> list[Path]:
    """Return immediate subdirectories of *parent* that contain a parquet."""
    if not parent.exists():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir())


def merge_parquets(shard_dirs: list[Path], output_path: Path) -> int:
    """Concatenate shard parquets, deduplicate, and write to *output_path*.

    Args:
        shard_dirs: Directories that may contain ``results_raw.parquet``.
        output_path: Destination parquet file.

    Returns:
        Total number of rows written after deduplication.
    """
    frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        p = shard_dir / "results_raw.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            print(f"  [skip] no parquet in {shard_dir.name}")

    if not frames:
        print("  No parquet shards found.")
        return 0

    combined = pd.concat(frames, ignore_index=True)

    # Keep latest row per run key to allow targeted re-runs to override stale
    # error rows.
    if "run_timestamp" in combined.columns:
        valid_keys = [k for k in _RUN_KEY if k in combined.columns]
        combined = (
            combined
            .sort_values("run_timestamp", ascending=True)
            .drop_duplicates(subset=valid_keys, keep="last")
            .reset_index(drop=True)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"  Written {len(combined)} rows → {output_path.relative_to(REPO_ROOT)}")
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

    print(f"  Written {total_lines} epoch log lines → {output_path.relative_to(REPO_ROOT)}")
    return total_lines


def main() -> None:
    """Merge all shards from long_term and m4 subdirectories."""
    print("=== Merging long-term results ===")
    lt_dir = RAW_DIR / "long_term"
    lt_shards = _shard_dirs(lt_dir)
    if lt_shards:
        merge_parquets(lt_shards, lt_dir / "results_raw.parquet")
        merge_jsonl(lt_shards, lt_dir / "epoch_logs.jsonl")
    else:
        print("  No long-term shards found.")

    print("\n=== Merging M4 results ===")
    m4_dir = RAW_DIR / "m4"
    m4_shards = _shard_dirs(m4_dir)
    if m4_shards:
        merge_parquets(m4_shards, m4_dir / "results_raw.parquet")
        merge_jsonl(m4_shards, m4_dir / "epoch_logs.jsonl")
    else:
        print("  No M4 shards found.")

    print("\n=== Merging all tracks into canonical results_raw.parquet ===")
    all_shards = lt_shards + m4_shards
    if all_shards:
        merge_parquets(all_shards, RAW_DIR / "results_raw.parquet")
        merge_jsonl(all_shards, RAW_DIR / "epoch_logs.jsonl")
    else:
        print("  No shards found anywhere.")

    print("\nDone.")


if __name__ == "__main__":
    main()
