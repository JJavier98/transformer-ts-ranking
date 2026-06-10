"""Publication-ready reporting for the transformer-ts-ranking benchmark.

This package reads persisted artifacts under ``results/`` and writes paper-
ready figures and tables to ``paper/figures/`` and ``paper/tables/``.  No
benchmark code is re-run; all outputs are derived purely from the raw parquet
and JSONL files produced by the benchmark runner.

Sub-modules:
    convergence  — per-epoch loss curves (PNG + PDF)
    radar        — accuracy × efficiency radar charts
    tables       — LaTeX and CSV ranking tables
    statistics   — Friedman test, Nemenyi post-hoc, CD diagram
    runner       — orchestrates the full report generation pipeline
"""

from .runner import ReportRunner, ReportRunnerConfig  # noqa: F401

__all__ = ["ReportRunner", "ReportRunnerConfig"]
