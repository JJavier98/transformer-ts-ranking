"""Orchestrates the full report generation pipeline.

Reads persisted benchmark artifacts (parquet + JSONL) and writes all
paper-ready outputs to ``paper/figures/`` and ``paper/tables/``.

No model training occurs here; all outputs are derived from the raw
``results/raw/results_raw.parquet`` and ``results/raw/epoch_logs.jsonl``
files written by the benchmark runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .convergence import load_epoch_logs, plot_convergence_curves
from .radar import generate_radar_charts
from .statistics import run_full_statistics
from .tables import generate_heatmaps, generate_ranking_tables


@dataclass
class ReportRunnerConfig:
    """Configuration for the report generation pipeline.

    Attributes:
        repo_root: Root of the benchmark repository.
        results_dir: Directory containing ``results_raw.parquet``
            and ``epoch_logs.jsonl``.
        figures_dir: Output directory for all figures.
        tables_dir: Output directory for all tables.
        stats_dir: Output directory for statistical test artifacts.
        tasks: Which benchmark tracks to include (``"long_term"`` / ``"m4"``).
    """

    repo_root: Path
    results_dir: Path | None = None
    figures_dir: Path | None = None
    tables_dir: Path | None = None
    stats_dir: Path | None = None
    tasks: list[str] = field(default_factory=lambda: ["long_term", "m4"])

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        if self.results_dir is None:
            self.results_dir = self.repo_root / "results" / "raw"
        if self.figures_dir is None:
            self.figures_dir = self.repo_root / "paper" / "figures"
        if self.tables_dir is None:
            self.tables_dir = self.repo_root / "paper" / "tables"
        if self.stats_dir is None:
            self.stats_dir = self.repo_root / "results" / "stats"


class ReportRunner:
    """Regenerates all paper-ready outputs from persisted artifacts.

    Usage::

        cfg = ReportRunnerConfig(repo_root=Path("."))
        runner = ReportRunner(cfg)
        summary = runner.run()
    """

    def __init__(self, config: ReportRunnerConfig) -> None:
        """Initialise the report runner.

        Args:
            config: Report generation configuration.
        """
        self.cfg = config

    def run(self) -> dict[str, Any]:
        """Generate all figures, tables, and statistical tests.

        Returns:
            Dict mapping output category to list of generated file paths.
        """
        results = self._load_results()
        if results.empty:
            print("No benchmark results found. Run the benchmark first.")
            return {}

        epoch_logs = self._load_epoch_logs()
        summary: dict[str, Any] = {}

        # 1 — Convergence curves
        if not epoch_logs.empty:
            print("Generating convergence curves...")
            paths = plot_convergence_curves(epoch_logs, self.cfg.figures_dir / "convergence")
            summary["convergence"] = [str(p) for p in paths]
            print(f"  → {len(paths)} files")

        # 2 — Accuracy heatmaps
        print("Generating accuracy heatmaps...")
        paths = generate_heatmaps(results, self.cfg.figures_dir)
        summary["heatmaps"] = [str(p) for p in paths]
        print(f"  → {len(paths)} files")

        # 3 — Radar charts
        print("Generating radar charts...")
        paths = generate_radar_charts(results, self.cfg.figures_dir)
        summary["radar"] = [str(p) for p in paths]
        print(f"  → {len(paths)} files")

        # 4 — Ranking tables
        print("Generating ranking tables...")
        paths = generate_ranking_tables(results, self.cfg.tables_dir)
        summary["tables"] = [str(p) for p in paths]
        print(f"  → {len(paths)} files")

        # 5 — Statistical tests + CD diagrams
        print("Running statistical tests...")
        stat_summary = run_full_statistics(
            results=results,
            stats_dir=self.cfg.stats_dir,
            figures_dir=self.cfg.figures_dir,
        )
        summary["statistics"] = stat_summary
        print(f"  → Friedman + Nemenyi + CD diagrams generated")

        return summary

    def _load_results(self) -> pd.DataFrame:
        """Load the raw results parquet or return an empty DataFrame."""
        parquet_path = self.cfg.results_dir / "results_raw.parquet"
        if not parquet_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(parquet_path)

    def _load_epoch_logs(self) -> pd.DataFrame:
        """Load the epoch log JSONL or return an empty DataFrame."""
        jsonl_path = self.cfg.results_dir / "epoch_logs.jsonl"
        if not jsonl_path.exists():
            return pd.DataFrame()
        return load_epoch_logs(jsonl_path)


