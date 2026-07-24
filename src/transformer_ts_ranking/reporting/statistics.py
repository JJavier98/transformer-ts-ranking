"""Statistical tests and Critical Difference (CD) diagram for the paper.

Implements the three-step significance analysis required by any ML benchmark
paper (Demšar, 2006):
  1. Friedman test — non-parametric omnibus test: are all models equivalent?
  2. Nemenyi post-hoc test — pairwise significant differences.
  3. CD diagram — visual summary of which pairwise differences exceed the
     critical difference at α=0.05.

Input:  ``results/raw/results_raw.parquet``
Output: ``results/stats/friedman_test.json``
        ``results/stats/nemenyi_matrix.csv``
        ``results/stats/cd_diagram_data.json``
        ``paper/figures/cd_diagram_{track}.{png,pdf}``

References:
  Demšar, J. (2006). Statistical comparisons of classifiers over multiple
  datasets. JMLR, 7, 1–30.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def _build_rank_matrix(
    results: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a model × dataset-horizon rank matrix for the Friedman test.

    Friedman and Nemenyi require a *complete* matrix.  When a model has not
    produced a value for every configuration, this drops the **incomplete
    models (rows)** — never the configurations (columns).  Dropping columns
    would let a single unfinished model erase whole dataset-horizon configs
    from the analysis: in the limit, one model that failed everywhere would
    empty the matrix and make the test impossible.  Dropping rows keeps all
    configurations and confines the loss to the models that genuinely could
    not run.

    This filtering is **in-memory only** — nothing is removed from the
    persisted parquet.  Excluded models are returned so callers report them
    explicitly; they belong in the paper's coverage table as N/A with a
    justification, never silently omitted.

    Args:
        results: Raw results DataFrame.
        metric: Metric to rank on (lower = better).

    Returns:
        Tuple of (rank matrix with models as rows and dataset-horizon configs
        as columns, where 1 = best model for that config; mapping of excluded
        model name -> number of configurations it was missing).
    """
    agg = (
        results
        .groupby(["model_name", "dataset_name", "horizon"])[metric]
        .mean()
        .reset_index()
    )
    agg["config"] = agg["dataset_name"] + "_h" + agg["horizon"].astype(str)
    pivot = agg.pivot(index="model_name", columns="config", values=metric)

    # Identify models with incomplete coverage before dropping anything, so the
    # exclusion is reportable rather than silent.
    missing_per_model = pivot.isna().sum(axis=1)
    excluded = {
        str(model): int(n_missing)
        for model, n_missing in missing_per_model.items()
        if n_missing > 0
    }

    # Drop incomplete MODELS (rows), preserving every configuration.
    complete = pivot.dropna(axis=0)
    # Rank within each config (ascending: lower metric = rank 1)
    rank_matrix = complete.rank(axis=0, ascending=True)
    return rank_matrix, excluded


def run_friedman_test(
    results: pd.DataFrame,
    metric: str = "mae",
    task: str = "long_term",
) -> dict[str, Any]:
    """Run the Friedman test on model ranks across datasets.

    Args:
        results: Raw results DataFrame.
        metric: Metric to rank on.
        task: ``"long_term"`` or ``"m4"``.

    Returns:
        Dict with statistic, p-value, degrees of freedom, and mean ranks.
        Also ``excluded_models`` (model -> number of missing configurations):
        models left out of the test because their coverage is incomplete.  An
        empty mapping means every eligible model entered the ranking.
    """
    subset = results[results["task"] == task]
    if subset.empty:
        return {"error": f"No results for task '{task}'."}

    rank_matrix, excluded = _build_rank_matrix(subset, metric)
    if excluded:
        print(
            f"[friedman:{task}:{metric}] WARNING — {len(excluded)} model(s) "
            f"excluded for incomplete coverage: {excluded}.  Complete their "
            f"missing cells and re-run to include them in the ranking."
        )
    if rank_matrix.shape[0] < 2 or rank_matrix.shape[1] < 2:
        return {
            "error": "Insufficient data for Friedman test (need ≥2 models and ≥2 configs).",
            "excluded_models": excluded,
        }

    # scipy.stats.friedmanchisquare expects one array per model (as rows)
    data = [rank_matrix.loc[m].values for m in rank_matrix.index]
    stat, p_value = stats.friedmanchisquare(*data)

    mean_ranks = rank_matrix.mean(axis=1).sort_values()

    return {
        "task": task,
        "metric": metric,
        "n_models": int(rank_matrix.shape[0]),
        "n_datasets": int(rank_matrix.shape[1]),
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
        "mean_ranks": mean_ranks.to_dict(),
        "excluded_models": excluded,
    }


# ---------------------------------------------------------------------------
# Nemenyi post-hoc test
# ---------------------------------------------------------------------------

def run_nemenyi_test(
    results: pd.DataFrame,
    metric: str = "mae",
    task: str = "long_term",
) -> pd.DataFrame:
    """Run the Nemenyi post-hoc test on model ranks.

    Args:
        results: Raw results DataFrame.
        metric: Metric to rank on.
        task: Benchmark track.

    Returns:
        Symmetric DataFrame of Nemenyi p-values (model × model).
    """
    subset = results[results["task"] == task]
    if subset.empty:
        return pd.DataFrame()

    rank_matrix, _excluded = _build_rank_matrix(subset, metric)
    if rank_matrix.shape[0] < 2:
        return pd.DataFrame()

    # scikit-posthocs expects an (n_datasets × n_models) DataFrame
    data_T = rank_matrix.T  # configs × models
    p_matrix = sp.posthoc_nemenyi_friedman(data_T)
    return p_matrix


# ---------------------------------------------------------------------------
# Critical Difference diagram
# ---------------------------------------------------------------------------

def _critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute the Nemenyi critical difference at a given α level.

    Uses the Studentised range distribution q-value from the look-up
    table in Demšar (2006) Table 5.  Values beyond k=20 use a linear
    interpolation of the provided table values.

    Args:
        n_models: Number of classifiers/models.
        n_datasets: Number of datasets (problems).
        alpha: Significance level.

    Returns:
        Critical difference threshold.
    """
    # q_{α, k} values from Demšar (2006) Table 5 for α=0.05 and α=0.10
    q_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        15: 3.394, 20: 3.561,
    }
    q_010 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
        7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
        15: 3.160, 20: 3.331,
    }
    table = q_005 if alpha <= 0.05 else q_010
    keys = sorted(table.keys())

    if n_models in table:
        q = table[n_models]
    elif n_models > max(keys):
        q = table[max(keys)]
    else:
        # Linear interpolation
        lo = max(k for k in keys if k < n_models)
        hi = min(k for k in keys if k > n_models)
        q = table[lo] + (table[hi] - table[lo]) * (n_models - lo) / (hi - lo)

    return float(q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets)))


def plot_cd_diagram(
    mean_ranks: dict[str, float],
    n_datasets: int,
    output_dir: Path,
    title: str = "CD Diagram",
    filename_stem: str = "cd_diagram",
    alpha: float = 0.05,
) -> list[Path]:
    """Draw a Demšar-style Critical Difference diagram.

    Groups of models whose mean-rank difference is below the CD are connected
    by a thick horizontal bar (they are NOT statistically significantly
    different).

    Args:
        mean_ranks: Dict mapping model names to their mean rank values.
        n_datasets: Number of datasets used in the Friedman test.
        output_dir: Output directory.
        title: Figure title.
        filename_stem: Base file name without extension.
        alpha: Significance level for the critical difference.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not mean_ranks:
        return []

    sorted_models = sorted(mean_ranks, key=mean_ranks.__getitem__)
    n_models = len(sorted_models)
    ranks_sorted = [mean_ranks[m] for m in sorted_models]
    cd = _critical_difference(n_models, n_datasets, alpha=alpha)

    # ---- Layout parameters --------------------------------------------------
    fig_width = 10
    fig_height = 0.5 * n_models + 2.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0.5, n_models + 0.5)
    ax.set_ylim(-0.5, n_models + 1.5)
    ax.axis("off")

    rank_min, rank_max = 1.0, float(n_models)
    axis_y = n_models + 0.8

    # Draw the rank axis
    ax.annotate(
        "", xy=(rank_max + 0.1, axis_y), xytext=(rank_min - 0.1, axis_y),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    for tick in np.arange(rank_min, rank_max + 0.5, 1.0):
        ax.plot([tick, tick], [axis_y - 0.1, axis_y + 0.1], "k-", lw=1.2)
        ax.text(tick, axis_y + 0.25, str(int(round(tick))), ha="center",
                fontsize=9, va="bottom")
    ax.text(
        (rank_min + rank_max) / 2, axis_y + 0.6,
        "← better   Mean rank   worse →",
        ha="center", fontsize=9, style="italic",
    )

    # Draw CD bracket
    cd_x_start = rank_min
    cd_x_end = rank_min + cd
    cd_y = axis_y + 1.2
    ax.annotate(
        "", xy=(cd_x_end, cd_y), xytext=(cd_x_start, cd_y),
        arrowprops=dict(arrowstyle="|-|", color="black", lw=2.0),
    )
    ax.text(
        (cd_x_start + cd_x_end) / 2, cd_y + 0.15,
        f"CD = {cd:.3f}",
        ha="center", fontsize=9,
    )

    # Draw model names and connecting lines
    for i, (model_name, rank) in enumerate(zip(sorted_models, ranks_sorted)):
        y = n_models - i - 0.5
        ax.plot([rank, rank], [axis_y, y + 0.4], "k-", lw=0.8, alpha=0.5)
        ax.plot(rank, axis_y, "ko", markersize=5)
        ha = "right" if rank <= (rank_min + rank_max) / 2 else "left"
        x_text = rank - 0.05 if ha == "right" else rank + 0.05
        ax.text(x_text, y + 0.3, model_name, ha=ha, va="center", fontsize=9)

    # Draw "no significant difference" clique bars
    _draw_clique_bars(ax, sorted_models, ranks_sorted, cd, axis_y)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()

    generated = []
    for ext in ("png", "pdf"):
        path = output_dir / f"{filename_stem}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        generated.append(path)
    plt.close(fig)
    return generated


def _draw_clique_bars(
    ax: plt.Axes,
    sorted_models: list[str],
    ranks: list[float],
    cd: float,
    axis_y: float,
) -> None:
    """Draw horizontal bars connecting models not significantly different.

    A clique is a maximal group of consecutive models (by rank) where the
    range of ranks is below the CD.  Each clique is represented by a thick
    horizontal bar at a y-position below the rank axis.

    Args:
        ax: Matplotlib axes.
        sorted_models: Model names sorted by ascending mean rank.
        ranks: Corresponding mean ranks.
        cd: Critical difference threshold.
        axis_y: Y-coordinate of the rank axis.
    """
    n = len(ranks)
    used = [False] * n
    bar_level = 0

    for i in range(n):
        if used[i]:
            continue
        # Find the maximal clique starting at i
        j = i
        while j + 1 < n and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:
            # Draw a bar from ranks[i] to ranks[j]
            bar_y = axis_y - 0.35 - bar_level * 0.25
            ax.plot([ranks[i], ranks[j]], [bar_y, bar_y], "k-", lw=3.5, alpha=0.75)
            for k in range(i, j + 1):
                used[k] = True
            bar_level += 1


# ---------------------------------------------------------------------------
# Full statistics pipeline
# ---------------------------------------------------------------------------

def run_full_statistics(
    results: pd.DataFrame,
    stats_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Run Friedman + Nemenyi tests and generate CD diagrams for both tracks.

    Args:
        results: Raw results DataFrame.
        stats_dir: Output directory for JSON/CSV stat artifacts.
        figures_dir: Output directory for CD diagram figures.

    Returns:
        Dict summarising all test results.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    for task, metric in [("long_term", "mae"), ("m4", "owa")]:
        subset = results[results["task"] == task]
        if subset.empty:
            continue

        # 1 — Friedman test
        friedman = run_friedman_test(results, metric=metric, task=task)
        summary[f"friedman_{task}"] = friedman
        (stats_dir / f"friedman_{task}.json").write_text(json.dumps(friedman, indent=2))

        if "error" in friedman:
            continue

        mean_ranks: dict[str, float] = friedman["mean_ranks"]
        n_datasets = int(friedman["n_datasets"])

        # 2 — Nemenyi post-hoc
        nemenyi_df = run_nemenyi_test(results, metric=metric, task=task)
        if not nemenyi_df.empty:
            nemenyi_path = stats_dir / f"nemenyi_matrix_{task}.csv"
            nemenyi_df.to_csv(nemenyi_path)
            summary[f"nemenyi_{task}"] = str(nemenyi_path)

        # 3 — CD diagram data
        cd_data: dict[str, Any] = {
            "task": task,
            "metric": metric,
            "n_datasets": n_datasets,
            "mean_ranks": mean_ranks,
            "cd_0_05": _critical_difference(len(mean_ranks), n_datasets, alpha=0.05),
            # Models absent from this diagram because their coverage was
            # incomplete — recorded so the figure's provenance is auditable.
            "excluded_models": friedman.get("excluded_models", {}),
        }
        (stats_dir / f"cd_diagram_data_{task}.json").write_text(
            json.dumps(cd_data, indent=2)
        )

        # 4 — CD diagram figure
        plot_cd_diagram(
            mean_ranks=mean_ranks,
            n_datasets=n_datasets,
            output_dir=figures_dir,
            title=f"CD Diagram — {task.replace('_', '-').title()} ({metric.upper()})",
            filename_stem=f"cd_diagram_{task}",
            alpha=0.05,
        )

    return summary
