"""Publication-ready reporting for the transformer-ts-ranking benchmark.

This package reads the persisted result shards under ``results/raw/`` and
writes paper-ready figures and tables to ``paper/figures/`` and
``paper/tables/`` (CSV copies in ``results/leaderboards/``).  No benchmark
code is re-run; all outputs are derived purely from the parquet and JSONL
files produced by the benchmark runner.

Sub-modules:
    loading      — shard loading, de-duplication, report scope, coverage
    tables       — LaTeX and CSV tables
    statistics   — Friedman (+ Iman–Davenport, Kendall's W), Nemenyi,
                   Wilcoxon–Holm vs. best, CD diagrams
    figures      — rank distributions, heatmaps, horizon scaling, M4,
                   convergence, accuracy vs. footprint
    convergence  — legacy per-run loss curves (not used by the runner)
    radar        — accuracy × efficiency radar charts (needs the
                   fixed-hardware profiling; not used by the runner yet)
    runner       — orchestrates the pipeline and writes the manifest
"""

from .runner import ReportRunner, ReportRunnerConfig  # noqa: F401

__all__ = ["ReportRunner", "ReportRunnerConfig"]
