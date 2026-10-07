"""Load persisted result shards and apply the paper's report scope.

Every SBATCH task writes its own shard ``results/raw/<track>/<shard>/
results_raw.parquet``.  This module is the single entry point through which
the reporting pipeline reads them (invariant: results are regenerated from
persisted artifacts only).  It:

1. Concatenates every shard of a track, remembering where each row came from.
2. Resolves duplicate runs of the same (model, dataset, horizon, seed): the
   most recent *successful* row wins; an error row is kept only when the cell
   never succeeded.  (A later out-of-memory retry must not erase an earlier
   valid result; a deliberate re-run that succeeds replaces the old row.)
3. Splits long-term rows into the core scope (main ranking) and the
   supplementary scope, and drops models excluded from the ranking
   (``configs/benchmark/report_scope.yaml``), keeping them for the coverage
   table.
4. Aggregates seeds per cell (mean and standard deviation over the seeds that
   succeeded) — metrics are averaged across seeds, never across datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RUN_KEY = ["model_name", "dataset_name", "horizon", "seed", "task"]
CELL_KEY = ["model_name", "dataset_name", "horizon"]
LT_METRICS = ["mae", "mse", "rmse"]
M4_METRICS = ["smape", "mase", "owa"]
EFFICIENCY_COLUMNS = [
    "train_time_s", "inference_latency_ms", "peak_gpu_mb", "param_count", "epochs_trained",
]


def load_scope(path: Path) -> dict[str, Any]:
    """Read the report-scope YAML.

    Args:
        path: Path to ``configs/benchmark/report_scope.yaml``.

    Returns:
        Parsed scope dictionary.
    """
    return yaml.safe_load(path.read_text())


def load_shards(track_dir: Path) -> pd.DataFrame:
    """Concatenate every ``<shard>/results_raw.parquet`` under a track directory.

    Aggregated files written directly in ``track_dir`` (e.g. by
    ``merge_results.py``) are ignored so no row is counted twice.

    Args:
        track_dir: ``results/raw/long_term`` or ``results/raw/m4``.

    Returns:
        All rows, with an extra ``shard`` column naming the source directory.
    """
    frames = []
    for path in sorted(track_dir.glob("*/results_raw.parquet")):
        frame = pd.read_parquet(path)
        frame["shard"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def is_ok(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of successful rows (no error message)."""
    return frame["error"].isna() | (frame["error"].astype(str).str.strip() == "")


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per run key: latest successful row, else latest error row.

    Args:
        frame: Concatenated shard rows.

    Returns:
        De-duplicated rows with an ``ok`` boolean column.
    """
    out = frame.copy()
    out["ok"] = is_ok(out)
    out["_ts"] = pd.to_datetime(out["run_timestamp"], format="ISO8601", utc=True)
    out = out.sort_values(["ok", "_ts"])  # successful and recent rows last
    out = out.drop_duplicates(subset=RUN_KEY, keep="last").drop(columns="_ts")
    return out.reset_index(drop=True)


def aggregate_seeds(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and standard deviation of each metric over the successful seeds of a cell.

    Args:
        frame: De-duplicated successful rows.
        metrics: Metric columns to aggregate.

    Returns:
        One row per (model, dataset, horizon) with ``<metric>`` (mean),
        ``<metric>_std`` and ``n_seeds`` / ``seeds`` columns.
    """
    grouped = frame.groupby(CELL_KEY)
    mean = grouped[metrics].mean()
    std = grouped[metrics].std(ddof=1).add_suffix("_std")
    seeds = grouped["seed"].agg(
        n_seeds="nunique", seeds=lambda s: ",".join(str(int(v)) for v in sorted(s.unique()))
    )
    return pd.concat([mean, std, seeds], axis=1).reset_index()


@dataclass
class ReportData:
    """Everything the tables, figures and tests need, already scoped.

    Attributes:
        scope: The applied report scope.
        lt_rows: De-duplicated long-term rows (all datasets, all models).
        m4_rows: De-duplicated M4 rows (all models).
        lt_core: Seed-aggregated long-term cells, core datasets, ranked models.
        lt_supp: Seed-aggregated long-term cells, supplementary datasets.
        m4_cells: Seed-aggregated M4 cells (one per model × frequency).
        ranked_models: Models that enter the main ranking.
        excluded: Excluded model -> {status, reason}.
        warnings: Human-readable data-quality warnings.
    """

    scope: dict[str, Any]
    lt_rows: pd.DataFrame
    m4_rows: pd.DataFrame
    lt_core: pd.DataFrame
    lt_supp: pd.DataFrame
    m4_cells: pd.DataFrame
    ranked_models: list[str]
    excluded: dict[str, dict[str, str]]
    warnings: list[str] = field(default_factory=list)

    def display(self, model: str) -> str:
        """Paper display name of a model (falls back to the canonical key)."""
        return self.scope.get("display_names", {}).get(model, model)

    def display_dataset(self, dataset: str) -> str:
        """Paper display name of a dataset."""
        return self.scope.get("dataset_display_names", {}).get(dataset, dataset)

    def expected_horizons(self, dataset: str) -> list[int]:
        """Protocol horizons of a long-term dataset."""
        horizons = self.scope["long_term"]["horizons"]
        return list(horizons.get(dataset, horizons["default"]))


def build_report_data(raw_dir: Path, scope_path: Path) -> ReportData:
    """Load all shards, de-duplicate them and apply the report scope.

    Args:
        raw_dir: ``results/raw``.
        scope_path: ``configs/benchmark/report_scope.yaml``.

    Returns:
        Scoped :class:`ReportData`.
    """
    scope = load_scope(scope_path)
    excluded = scope.get("excluded_models", {})
    warnings: list[str] = []

    lt_rows = load_shards(raw_dir / "long_term")
    m4_rows = load_shards(raw_dir / "m4")
    lt_rows = deduplicate(lt_rows) if not lt_rows.empty else lt_rows
    m4_rows = deduplicate(m4_rows) if not m4_rows.empty else m4_rows

    for name, rows in (("long_term", lt_rows), ("m4", m4_rows)):
        if not rows.empty and "precision" in rows:
            bad = rows.loc[rows["ok"] & (rows["precision"] != "fp32")]
            if not bad.empty:
                warnings.append(
                    f"{name}: {len(bad)} successful rows are not fp32 "
                    f"({sorted(bad['model_name'].unique())})."
                )

    core = scope["long_term"]["core_datasets"]
    supp = scope["long_term"]["supplementary_datasets"]
    lt_ok = lt_rows[lt_rows["ok"] & ~lt_rows["model_name"].isin(excluded)]
    m4_ok = m4_rows[m4_rows["ok"] & ~m4_rows["model_name"].isin(excluded)]

    lt_core = aggregate_seeds(lt_ok[lt_ok["dataset_name"].isin(core)], LT_METRICS)
    lt_supp = aggregate_seeds(lt_ok[lt_ok["dataset_name"].isin(supp)], LT_METRICS)
    m4_cells = aggregate_seeds(m4_ok, M4_METRICS) if not m4_ok.empty else pd.DataFrame()

    ranked = sorted(set(lt_core["model_name"]) | set(m4_cells.get("model_name", [])))
    return ReportData(
        scope=scope,
        lt_rows=lt_rows,
        m4_rows=m4_rows,
        lt_core=lt_core,
        lt_supp=lt_supp,
        m4_cells=m4_cells,
        ranked_models=ranked,
        excluded=excluded,
        warnings=warnings,
    )


def coverage_table(data: ReportData) -> pd.DataFrame:
    """Per-model coverage of the protocol grid, per track and scope.

    A long-term cell counts as covered when at least one seed succeeded; the
    number of successful seeds and how many replacement seeds were used are
    reported alongside.

    Args:
        data: Scoped report data.

    Returns:
        One row per model (ranked and excluded) with covered/expected cells
        for the core datasets, the supplementary datasets and M4.
    """
    scope = data.scope
    core = scope["long_term"]["core_datasets"]
    supp = scope["long_term"]["supplementary_datasets"]
    protocol = set(scope["seeds"]["protocol"])
    models = sorted(set(data.lt_rows.get("model_name", [])) | set(data.m4_rows.get("model_name", [])))
    lt_ok = data.lt_rows[data.lt_rows["ok"]]
    m4_ok = data.m4_rows[data.m4_rows["ok"]] if not data.m4_rows.empty else data.m4_rows

    rows = []
    for model in models:
        row: dict[str, Any] = {"model": model, "display": data.display(model)}
        for label, datasets in (("core", core), ("supplementary", supp)):
            expected = sum(len(data.expected_horizons(d)) for d in datasets)
            sub = lt_ok[(lt_ok["model_name"] == model) & lt_ok["dataset_name"].isin(datasets)]
            cells = sub.groupby(["dataset_name", "horizon"])["seed"].nunique()
            row[f"{label}_cells"] = f"{len(cells)}/{expected}"
            row[f"{label}_runs"] = int(len(sub))
            if label == "core":
                row["core_replacement_seed_runs"] = int((~sub["seed"].isin(protocol)).sum())
                row["core_min_seeds_per_cell"] = int(cells.min()) if len(cells) else 0
        n_freq = len(scope["m4"]["frequencies"])
        m4_model = m4_ok[m4_ok["model_name"] == model] if not m4_ok.empty else m4_ok
        row["m4_cells"] = f"{m4_model['dataset_name'].nunique() if len(m4_model) else 0}/{n_freq}"
        status = data.excluded.get(model, {}).get("status", "ranked")
        row["status"] = status
        rows.append(row)
    return pd.DataFrame(rows)
