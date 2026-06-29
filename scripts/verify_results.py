#!/usr/bin/env python
"""Post-run verification of the benchmark results.

Validates the persisted parquet artifacts **only** (CLAUDE.md invariant #5:
results are regenerated from raw artifacts, never from memory or a live run).
It answers the questions that gate a publishable ranking:

  1. **Coverage** — does every expected cell have an error-free row?
     Long-term cell = (model, dataset, horizon, seed).
     M4 cell        = (model, frequency, seed).
     Expected models/datasets/horizons/seeds are read from the manifests, so
     the check stays in sync with the capability matrix automatically.

  2. **Precision uniformity** — do all rows share one precision (default
     ``fp32``)?  Mixing bf16 and fp32 shards breaks cross-shard comparability,
     so any deviation is flagged as an error.

  3. **Dedup integrity** — is there exactly one row per run key?  Duplicates
     would mean ``_append_to_parquet`` dedup failed.

  4. **Error inventory** — which cells still carry an error, grouped by model
     and message, so reruns can target them.

  5. **Model-set consistency** — does every dataset see the same set of models?

Usage::

    .venv/bin/python scripts/verify_results.py
    .venv/bin/python scripts/verify_results.py --expected-precision fp32
    .venv/bin/python scripts/verify_results.py --results-dir results/raw --json report.json

Exit code: ``0`` when coverage is complete, precision uniform, and no
duplicates; ``1`` when any gap, error, or inconsistency is found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "benchmark"
DEFAULT_SEEDS = [42, 123, 2026]


# ---------------------------------------------------------------------------
# Expected coverage (from manifests)
# ---------------------------------------------------------------------------
def load_eligible_models(track: str) -> set[str]:
    """Return the set of model keys eligible for ``track`` ('long_term'|'m4').

    Reads the capability matrix and honours the ``eligible_<track>`` flag so
    permanently-excluded models (e.g. tpatchgnn) are never counted as missing.
    """
    matrix = yaml.safe_load((CONFIG_DIR / "model_capability_matrix.yaml").read_text())
    flag = f"eligible_{track}"
    return {
        m["model_name"]
        for m in matrix["models"]
        if m.get(flag, False)
    }


def load_long_term_expectations() -> dict[str, list[int]]:
    """Return ``{dataset_name: [horizons]}`` for the long-term track."""
    manifest = yaml.safe_load((CONFIG_DIR / "long_term_datasets.yaml").read_text())
    return {name: list(spec["horizons"]) for name, spec in manifest["datasets"].items()}


def load_m4_frequencies() -> list[str]:
    """Return the list of M4 frequency slices."""
    manifest = yaml.safe_load((CONFIG_DIR / "m4_datasets.yaml").read_text())
    return list(manifest["frequencies"].keys())


# ---------------------------------------------------------------------------
# Actual coverage (from parquet)
# ---------------------------------------------------------------------------
def load_track_frame(results_dir: Path, track: str) -> pd.DataFrame:
    """Concatenate every ``results_raw.parquet`` shard for ``track``.

    Returns an empty frame (with no rows) if no shard exists yet.
    """
    sub = "long_term" if track == "long_term" else "m4"
    frames = [
        pd.read_parquet(p)
        for p in sorted((results_dir / sub).glob("*/results_raw.parquet"))
    ]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # A 'precision' column may be absent in shards written before the column
    # was introduced; mark those rows so the precision check can report them.
    if "precision" not in df.columns:
        df["precision"] = "unknown"
    else:
        df["precision"] = df["precision"].fillna("unknown")
    return df


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_long_term(df: pd.DataFrame, models: set[str], datasets: dict[str, list[int]],
                    seeds: list[int]) -> dict[str, Any]:
    """Return a coverage report for the long-term track."""
    ok = df[df["error"].isna()] if not df.empty else df
    present = set(
        zip(ok["model_name"], ok["dataset_name"], ok["horizon"].astype(int), ok["seed"].astype(int))
    ) if not ok.empty else set()

    expected = {
        (model, ds, h, seed)
        for model in models
        for ds, horizons in datasets.items()
        for h in horizons
        for seed in seeds
    }
    missing = expected - present
    return {
        "expected": len(expected),
        "present": len(expected & present),
        "missing": sorted(missing),
    }


def check_m4(df: pd.DataFrame, models: set[str], freqs: list[str],
             seeds: list[int]) -> dict[str, Any]:
    """Return a coverage report for the M4 track (no horizon dimension)."""
    ok = df[df["error"].isna()] if not df.empty else df
    present = set(
        zip(ok["model_name"], ok["dataset_name"], ok["seed"].astype(int))
    ) if not ok.empty else set()

    expected = {
        (model, freq, seed)
        for model in models
        for freq in freqs
        for seed in seeds
    }
    missing = expected - present
    return {
        "expected": len(expected),
        "present": len(expected & present),
        "missing": sorted(missing),
    }


def check_precision(df: pd.DataFrame, expected: str) -> dict[str, Any]:
    """Return the precision distribution and any rows deviating from ``expected``."""
    if df.empty:
        return {"counts": {}, "deviating": 0, "uniform": True}
    counts = df["precision"].value_counts().to_dict()
    deviating = df[df["precision"] != expected]
    bad = deviating.groupby(["dataset_name", "precision"]).size().to_dict() if not deviating.empty else {}
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "deviating": int(len(deviating)),
        "by_dataset_precision": {f"{k[0]}|{k[1]}": int(v) for k, v in bad.items()},
        "uniform": bool(len(deviating) == 0),
    }


def check_duplicates(df: pd.DataFrame, key: list[str]) -> list[tuple]:
    """Return run keys that appear more than once (dedup integrity)."""
    if df.empty:
        return []
    dup = df.groupby(key).size()
    return sorted(tuple(k) for k, n in dup.items() if n > 1)


def check_errors(df: pd.DataFrame) -> dict[str, Any]:
    """Return an inventory of error rows grouped by model and short message."""
    if df.empty:
        return {"total": 0, "by_model": {}}
    err = df[df["error"].notna()]
    by_model = {}
    for model, grp in err.groupby("model_name"):
        msg = str(grp["error"].iloc[0]).splitlines()[0][:80]
        by_model[model] = {"count": int(len(grp)), "sample": msg}
    return {"total": int(len(err)), "by_model": by_model}


def check_model_set_consistency(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return, per dataset, models that are absent relative to the union seen."""
    if df.empty:
        return {}
    seen_by_ds = {ds: set(grp["model_name"]) for ds, grp in df.groupby("dataset_name")}
    union = set().union(*seen_by_ds.values()) if seen_by_ds else set()
    return {
        ds: sorted(union - seen)
        for ds, seen in seen_by_ds.items()
        if union - seen
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "raw")
    parser.add_argument("--expected-precision", default="fp32",
                        help="Precision every row must share (default: fp32).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to write the full report as JSON.")
    parser.add_argument("--max-list", type=int, default=20,
                        help="Max missing/error entries to print per section.")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    problems = 0

    # ---- Long-term -------------------------------------------------------
    lt_models = load_eligible_models("long_term")
    lt_datasets = load_long_term_expectations()
    lt_df = load_track_frame(args.results_dir, "long_term")
    lt_cov = check_long_term(lt_df, lt_models, lt_datasets, args.seeds)
    report["long_term_coverage"] = {**lt_cov, "missing": lt_cov["missing"][: args.max_list]}

    _print_section("LONG-TERM COVERAGE")
    print(f"models={len(lt_models)}  datasets={len(lt_datasets)}  seeds={args.seeds}")
    print(f"cells present (OK): {lt_cov['present']} / {lt_cov['expected']}")
    if lt_cov["missing"]:
        problems += 1
        print(f"MISSING {len(lt_cov['missing'])} cells (showing {args.max_list}):")
        for cell in lt_cov["missing"][: args.max_list]:
            print(f"  - {cell}")
    else:
        print("complete.")

    # ---- M4 --------------------------------------------------------------
    m4_models = load_eligible_models("m4")
    m4_freqs = load_m4_frequencies()
    m4_df = load_track_frame(args.results_dir, "m4")
    m4_cov = check_m4(m4_df, m4_models, m4_freqs, args.seeds)
    report["m4_coverage"] = {**m4_cov, "missing": m4_cov["missing"][: args.max_list]}

    _print_section("M4 COVERAGE")
    print(f"models={len(m4_models)}  frequencies={len(m4_freqs)}  seeds={args.seeds}")
    print(f"cells present (OK): {m4_cov['present']} / {m4_cov['expected']}")
    if m4_cov["missing"]:
        problems += 1
        print(f"MISSING {len(m4_cov['missing'])} cells (showing {args.max_list}):")
        for cell in m4_cov["missing"][: args.max_list]:
            print(f"  - {cell}")
    else:
        print("complete.")

    # ---- Precision -------------------------------------------------------
    all_df = pd.concat([d for d in (lt_df, m4_df) if not d.empty], ignore_index=True) \
        if (not lt_df.empty or not m4_df.empty) else pd.DataFrame()
    prec = check_precision(all_df, args.expected_precision)
    report["precision"] = prec

    _print_section(f"PRECISION UNIFORMITY (expected: {args.expected_precision})")
    print(f"distribution: {prec['counts']}")
    if not prec["uniform"]:
        problems += 1
        print(f"NON-UNIFORM: {prec['deviating']} rows deviate from {args.expected_precision}")
        for k, v in prec.get("by_dataset_precision", {}).items():
            print(f"  - {k}: {v} rows")
    else:
        print("uniform.")

    # ---- Duplicates ------------------------------------------------------
    lt_dup = check_duplicates(lt_df, ["model_name", "dataset_name", "horizon", "seed", "task"])
    m4_dup = check_duplicates(m4_df, ["model_name", "dataset_name", "seed", "task"])
    report["duplicates"] = {"long_term": len(lt_dup), "m4": len(m4_dup)}

    _print_section("DEDUP INTEGRITY")
    if lt_dup or m4_dup:
        problems += 1
        print(f"DUPLICATE run keys — long_term={len(lt_dup)} m4={len(m4_dup)}")
        for cell in (lt_dup + m4_dup)[: args.max_list]:
            print(f"  - {cell}")
    else:
        print("one row per run key — OK.")

    # ---- Errors ----------------------------------------------------------
    _print_section("ERROR INVENTORY (informational — reruns retry these)")
    for track, df in (("long_term", lt_df), ("m4", m4_df)):
        errs = check_errors(df)
        report[f"{track}_errors"] = errs
        print(f"[{track}] error rows: {errs['total']}")
        for model, info in sorted(errs["by_model"].items()):
            print(f"  - {model}: {info['count']}  ({info['sample']})")

    # ---- Model-set consistency ------------------------------------------
    _print_section("MODEL-SET CONSISTENCY (per dataset, models absent vs union)")
    incons = {}
    for track, df in (("long_term", lt_df), ("m4", m4_df)):
        c = check_model_set_consistency(df)
        if c:
            incons[track] = c
            for ds, miss in c.items():
                print(f"  [{track}] {ds}: missing {miss}")
    if not incons:
        print("consistent (every dataset saw the same model set).")
    report["model_set_inconsistency"] = incons

    # ---- Summary ---------------------------------------------------------
    _print_section("SUMMARY")
    verdict = "PASS" if problems == 0 else f"ISSUES ({problems} categories)"
    print(f"verdict: {verdict}")
    print("note: error rows are informational — they are retried automatically "
          "on resubmission and do not by themselves fail this check.")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nfull report written to {args.json}")

    # Coverage gaps, non-uniform precision, and duplicates are hard failures.
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
