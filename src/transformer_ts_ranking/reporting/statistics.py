"""Statistical tests and Critical Difference (CD) diagrams for the paper.

Implements the significance analysis of Demšar (2006) on the seed-averaged
cells produced by :mod:`.loading`.  A *block* is one (dataset, horizon)
configuration (or one M4 frequency); models are ranked within each block
(1 = lowest error, ties share the average rank).

1. Friedman test, with the Iman–Davenport F correction and Kendall's W as
   effect size (W = χ²_F / (N (k − 1)); 0 = no agreement between blocks,
   1 = identical ranking in every block).
2. Nemenyi post-hoc test (all pairs) and the critical difference
   CD = q_α √(k (k + 1) / (6 N)), with q_α the studentised-range quantile
   divided by √2 — computed exactly for any k (no table look-up).
3. Wilcoxon signed-rank tests of every model against the best-ranked one,
   Holm-corrected, with the matched-pairs rank-biserial correlation as effect
   size (Benavoli et al., 2016 recommend this over Nemenyi for pairwise
   claims).
4. Demšar-style CD diagram.

Friedman / Nemenyi need a complete matrix: models without a value in every
block are left out of the test (never the blocks) and reported as excluded.

References:
  Demšar, J. (2006). Statistical comparisons of classifiers over multiple
  data sets. JMLR 7, 1–30.
  Benavoli, A., Corani, G., Mangili, F. (2016). Should we really use post-hoc
  tests based on mean-ranks? JMLR 17, 1–10.
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
# Rank matrix
# ---------------------------------------------------------------------------

def build_block_matrix(cells: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Model × block matrix of a metric, restricted to complete models.

    Args:
        cells: Seed-aggregated cells with ``model_name``, ``dataset_name``,
            ``horizon`` and the metric column.
        metric: Metric to use (lower is better).

    Returns:
        ``(values, excluded)``: values with models as rows and
        ``<dataset>_h<horizon>`` blocks as columns (complete models only),
        and excluded model -> number of missing blocks.
    """
    frame = cells.copy()
    frame["block"] = frame["dataset_name"] + "_h" + frame["horizon"].astype(str)
    pivot = frame.pivot_table(index="model_name", columns="block", values=metric)
    missing = pivot.isna().sum(axis=1)
    excluded = {str(m): int(n) for m, n in missing.items() if n > 0}
    return pivot.dropna(axis=0), excluded


def rank_matrix(values: pd.DataFrame) -> pd.DataFrame:
    """Rank models within each block (1 = best, ties averaged)."""
    return values.rank(axis=0, ascending=True, method="average")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def critical_difference(n_models: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for ``n_models`` over ``n_blocks``.

    Args:
        n_models: Number of compared models (k).
        n_blocks: Number of blocks (N).
        alpha: Significance level.

    Returns:
        CD = q_α √(k (k + 1) / (6 N)), q_α = studentised range(1−α; k, ∞) / √2.
    """
    q_alpha = stats.studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_blocks)))


def friedman_test(values: pd.DataFrame) -> dict[str, Any]:
    """Friedman test with Iman–Davenport correction and Kendall's W.

    Args:
        values: Complete model × block matrix (lower is better).

    Returns:
        Test summary (statistics, p-values, effect size, mean ranks).
    """
    k, n = values.shape
    ranks = rank_matrix(values)
    chi2, p_chi2 = stats.friedmanchisquare(*[values.loc[m].to_numpy() for m in values.index])
    # Iman & Davenport (1980): less conservative F version of the statistic.
    f_stat = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    kendall_w = chi2 / (n * (k - 1))
    return {
        "n_models": int(k),
        "n_blocks": int(n),
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_chi2),
        "iman_davenport_F": float(f_stat),
        "iman_davenport_p": p_f,
        "kendall_W": float(kendall_w),
        "significant_at_0_05": bool(p_f < 0.05),
        "mean_ranks": ranks.mean(axis=1).sort_values().to_dict(),
    }


def nemenyi_matrix(values: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Nemenyi p-values (model × model)."""
    return sp.posthoc_nemenyi_friedman(values.T.reset_index(drop=True))


def wilcoxon_vs_best(values: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test of every model against the best-ranked model.

    Args:
        values: Complete model × block matrix (lower is better).

    Returns:
        One row per model: mean rank, Holm-adjusted p-value and the
        matched-pairs rank-biserial correlation r (positive = the model has
        larger errors than the best model).
    """
    mean_ranks = rank_matrix(values).mean(axis=1).sort_values()
    best = mean_ranks.index[0]
    rows = []
    for model in mean_ranks.index[1:]:
        diff = values.loc[model] - values.loc[best]
        nonzero = diff[diff != 0]
        if len(nonzero) == 0:
            p, r = 1.0, 0.0
        else:
            p = float(stats.wilcoxon(nonzero, zero_method="wilcox").pvalue)
            ranks = stats.rankdata(np.abs(nonzero))
            r = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
        rows.append({"model": model, "mean_rank": mean_ranks[model], "p_raw": p,
                     "rank_biserial_r": r})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Holm step-down correction.
    order = np.argsort(table["p_raw"].to_numpy())
    m = len(order)
    adjusted = np.empty(m)
    running = 0.0
    for pos, idx in enumerate(order):
        running = max(running, min(1.0, (m - pos) * table["p_raw"].iloc[idx]))
        adjusted[idx] = running
    table["p_holm"] = adjusted
    table["significant_at_0_05"] = table["p_holm"] < 0.05
    table.insert(0, "reference", best)
    return table


# ---------------------------------------------------------------------------
# CD diagram
# ---------------------------------------------------------------------------

def _cliques(ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Maximal groups of consecutive models whose rank range is below the CD."""
    n = len(ranks)
    cliques: list[tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    return cliques


def plot_cd_diagram(
    mean_ranks: dict[str, float],
    n_blocks: int,
    output_stem: Path,
    title: str,
    labels: dict[str, str] | None = None,
    alpha: float = 0.05,
) -> list[Path]:
    """Draw a Demšar CD diagram (better half on the left, worse half on the right).

    Args:
        mean_ranks: Model -> mean rank.
        n_blocks: Number of blocks used for the ranks.
        output_stem: Output path without extension (``.png`` and ``.pdf``).
        title: Figure title.
        labels: Optional model -> display label.
        alpha: Significance level of the CD.

    Returns:
        Written file paths.
    """
    labels = labels or {}
    models = sorted(mean_ranks, key=mean_ranks.__getitem__)
    ranks = [mean_ranks[m] for m in models]
    k = len(models)
    cd = critical_difference(k, n_blocks, alpha)
    half = (k + 1) // 2
    cliques = _cliques(ranks, cd)

    line_h = 0.28
    height = 1.6 + line_h * max(half, k - half) + 0.18 * len(cliques)
    fig, ax = plt.subplots(figsize=(10, height))
    lo, hi = 1, k
    ax.set_xlim(lo - (hi - lo) * 0.42, hi + (hi - lo) * 0.42)
    ax.set_ylim(-height + 0.6, 0.9)
    ax.axis("off")

    # Rank axis with ticks.
    ax.plot([lo, hi], [0, 0], "k-", lw=1)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [0, 0.08], "k-", lw=0.8)
        ax.text(tick, 0.13, str(tick), ha="center", va="bottom", fontsize=7)
    # CD bracket.
    ax.plot([lo, lo + cd], [0.55, 0.55], "k-", lw=1.4)
    ax.plot([lo, lo], [0.5, 0.6], "k-", lw=1.4)
    ax.plot([lo + cd, lo + cd], [0.5, 0.6], "k-", lw=1.4)
    ax.text(lo + cd / 2, 0.62, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

    # Clique bars just below the axis.
    for level, (i, j) in enumerate(cliques):
        y = -0.12 - 0.12 * level
        ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [y, y], "k-", lw=2.5, solid_capstyle="round")
    base = -0.25 - 0.12 * len(cliques)

    # Model labels with elbow connectors.
    for idx, (model, rank) in enumerate(zip(models, ranks)):
        left = idx < half
        row = idx if left else (k - 1 - idx)
        y = base - line_h * row
        x_text = lo - (hi - lo) * 0.05 if left else hi + (hi - lo) * 0.05
        ax.plot([rank, rank], [0, y], "k-", lw=0.6)
        ax.plot([rank, x_text], [y, y], "k-", lw=0.6)
        ax.text(x_text + (-0.05 if left else 0.05), y,
                f"{labels.get(model, model)} ({rank:.2f})",
                ha="right" if left else "left", va="center", fontsize=8)

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        path = output_stem.with_suffix(f".{ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_statistics(
    cells: pd.DataFrame,
    metric: str,
    name: str,
    stats_dir: Path,
    figures_dir: Path,
    labels: dict[str, str],
    title: str,
) -> dict[str, Any]:
    """Run every test for one (track, metric) and write the artifacts.

    Writes ``results/stats/{name}_friedman.json``, ``{name}_nemenyi.csv``,
    ``{name}_wilcoxon_vs_best.csv``, ``{name}_cd.json`` and the figure
    ``paper/figures/cd_{name}.{png,pdf}``.

    Args:
        cells: Seed-aggregated cells.
        metric: Metric column (lower is better).
        name: Artifact name prefix (e.g. ``long_term_mae``).
        stats_dir: Output directory for the statistics.
        figures_dir: Output directory for the CD diagram.
        labels: Model display labels.
        title: CD diagram title.

    Returns:
        The Friedman summary, including excluded models.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    values, excluded = build_block_matrix(cells, metric)
    if values.shape[0] < 3 or values.shape[1] < 2:
        summary = {"error": "not enough complete models/blocks", "excluded_models": excluded}
        (stats_dir / f"{name}_friedman.json").write_text(json.dumps(summary, indent=2))
        return summary

    summary = friedman_test(values)
    summary.update({"metric": metric, "excluded_models": excluded,
                    "blocks": list(values.columns)})
    cd = critical_difference(summary["n_models"], summary["n_blocks"])
    summary["nemenyi_cd_0_05"] = cd
    (stats_dir / f"{name}_friedman.json").write_text(json.dumps(summary, indent=2))

    nemenyi_matrix(values).set_axis(values.index, axis=0).set_axis(values.index, axis=1) \
        .to_csv(stats_dir / f"{name}_nemenyi.csv")
    wilcoxon_vs_best(values).to_csv(stats_dir / f"{name}_wilcoxon_vs_best.csv", index=False)
    rank_matrix(values).to_csv(stats_dir / f"{name}_ranks_per_block.csv")
    (stats_dir / f"{name}_cd.json").write_text(json.dumps({
        "mean_ranks": summary["mean_ranks"], "n_blocks": summary["n_blocks"],
        "cd_0_05": cd, "excluded_models": excluded,
    }, indent=2))
    plot_cd_diagram(summary["mean_ranks"], summary["n_blocks"],
                    figures_dir / f"cd_{name}", title, labels)
    return summary
