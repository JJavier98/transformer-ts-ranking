"""Orchestrates the full report generation pipeline.

Reads the persisted result shards (``results/raw/<track>/<shard>/``), applies
the report scope (``configs/benchmark/report_scope.yaml``) and writes:

* ``paper/tables/``        — LaTeX tables,
* ``paper/figures/``       — PNG + PDF figures,
* ``results/leaderboards/`` — CSV versions of every table,
* ``results/stats/``       — Friedman / Nemenyi / Wilcoxon artifacts,
* ``results/report_manifest.json`` — provenance: inputs, code versions,
  applied scope, ranked / excluded models, warnings and every output file.

No model is trained or evaluated here (invariant: results are regenerated
from persisted artifacts only).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .figures import generate_figures
from .loading import build_report_data
from .statistics import run_statistics
from .tables import generate_tables


@dataclass
class ReportRunnerConfig:
    """Paths used by the report pipeline.

    Attributes:
        repo_root: Root of the benchmark repository.
        results_dir: ``results/raw`` (per-track shard directories).
        scope_path: Report scope YAML.
        figures_dir: Output directory for figures.
        tables_dir: Output directory for LaTeX tables.
        leaderboards_dir: Output directory for CSV tables.
        stats_dir: Output directory for statistical artifacts.
    """

    repo_root: Path
    results_dir: Path | None = None
    scope_path: Path | None = None
    figures_dir: Path | None = None
    tables_dir: Path | None = None
    leaderboards_dir: Path | None = None
    stats_dir: Path | None = None

    def __post_init__(self) -> None:
        """Fill default paths relative to the repository root."""
        root = self.repo_root = self.repo_root.resolve()
        self.results_dir = self.results_dir or root / "results" / "raw"
        self.scope_path = self.scope_path or root / "configs" / "benchmark" / "report_scope.yaml"
        self.figures_dir = self.figures_dir or root / "paper" / "figures"
        self.tables_dir = self.tables_dir or root / "paper" / "tables"
        self.leaderboards_dir = self.leaderboards_dir or root / "results" / "leaderboards"
        self.stats_dir = self.stats_dir or root / "results" / "stats"


def _git(args: list[str], cwd: Path) -> str:
    """Return the output of a git command, or ``"unknown"``."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class ReportRunner:
    """Regenerates all paper-ready outputs from persisted artifacts."""

    def __init__(self, config: ReportRunnerConfig) -> None:
        """Store the configuration.

        Args:
            config: Report generation configuration.
        """
        self.cfg = config

    def run(self) -> dict[str, Any]:
        """Generate tables, statistics and figures, then write the manifest.

        Returns:
            The manifest dictionary.
        """
        cfg = self.cfg
        data = build_report_data(cfg.results_dir, cfg.scope_path)
        labels = {m: data.display(m).replace("$^\\dagger$", "†") for m in data.ranked_models}

        outputs: dict[str, list[str]] = {}
        print("Tables...")
        outputs["tables"] = [str(p) for p in generate_tables(data, cfg.tables_dir,
                                                             cfg.leaderboards_dir)]
        print("Statistics...")
        stats_summary = {}
        for metric in ("mae", "mse"):
            stats_summary[f"long_term_{metric}"] = run_statistics(
                data.lt_core, metric, f"long_term_{metric}", cfg.stats_dir, cfg.figures_dir,
                labels, f"Long-term forecasting — mean rank by {metric.upper()} "
                        f"(Nemenyi CD, α = 0.05)")
        if not data.m4_cells.empty:
            stats_summary["m4_owa"] = run_statistics(
                data.m4_cells, "owa", "m4_owa", cfg.stats_dir, cfg.figures_dir, labels,
                "M4 — mean rank by OWA over frequencies (Nemenyi CD, α = 0.05)")
        print("Figures...")
        outputs["figures"] = [str(p) for p in generate_figures(data, cfg.results_dir,
                                                               cfg.figures_dir)]

        lt_ok = data.lt_rows[data.lt_rows["ok"]]
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_commit": _git(["rev-parse", "HEAD"], cfg.repo_root),
            "repo_dirty": bool(_git(["status", "--porcelain"], cfg.repo_root)),
            "library_commit": _git(["rev-parse", "HEAD"], cfg.repo_root / "s-transformers-lib"),
            "scope_version": data.scope.get("version"),
            "inputs": {
                "long_term_shards": int(data.lt_rows["shard"].nunique()) if len(lt_ok) else 0,
                "long_term_runs_ok": int(len(lt_ok)),
                "long_term_runs_error": int((~data.lt_rows["ok"]).sum()),
                "m4_runs_ok": int(data.m4_rows["ok"].sum()) if not data.m4_rows.empty else 0,
                "m4_runs_error": int((~data.m4_rows["ok"]).sum()) if not data.m4_rows.empty else 0,
                "gpus": lt_ok["gpu_name"].dropna().value_counts().to_dict()
                if "gpu_name" in lt_ok else {},
            },
            "ranked_models": data.ranked_models,
            "excluded_models": data.excluded,
            "statistics": {k: {kk: v.get(kk) for kk in (
                "n_models", "n_blocks", "friedman_chi2", "friedman_p", "iman_davenport_F",
                "iman_davenport_p", "kendall_W", "nemenyi_cd_0_05", "excluded_models", "error")}
                for k, v in stats_summary.items()},
            "warnings": data.warnings,
            "pending": [
                "Training time and inference latency are not reported: past runs mixed V100 "
                "and A100 nodes. They come from the fixed-hardware profiling pass.",
            ],
            "outputs": outputs,
        }
        path = cfg.repo_root / "results" / "report_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"Manifest → {path}")
        return manifest
