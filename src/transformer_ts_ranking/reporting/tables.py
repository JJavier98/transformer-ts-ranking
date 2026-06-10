"""Ranking tables for long-term and M4 benchmark results.

Generates:
  - Accuracy heatmaps (model × dataset-horizon) as PNG/PDF.
  - Per-track ranking tables as CSV and LaTeX (``paper/tables/``).
  - An efficiency leaderboard ranking models by composite score.

Rank aggregation uses the mean of per-dataset-horizon ranks (average rank
across all evaluation configurations) rather than averaging raw metrics
across datasets of different scales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_rank(results: pd.DataFrame, metric: str) -> pd.Series:
    """Compute each model's mean rank across all (dataset, horizon) configs.

    Args:
        results: Raw results DataFrame.
        metric: Column name to rank on (lower = better assumed).

    Returns:
        Series indexed by model_name with mean rank values.
    """
    agg = results.groupby(["model_name", "dataset_name", "horizon"])[metric].mean().reset_index()
    agg["rank"] = agg.groupby(["dataset_name", "horizon"])[metric].rank(ascending=True)
    return agg.groupby("model_name")["rank"].mean().rename("mean_rank")


def _pivot_metrics(
    results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Pivot results into a model × dataset-horizon matrix of mean metric values.

    Args:
        results: Raw results DataFrame.
        metric: Metric column to pivot.

    Returns:
        DataFrame with models as rows and ``dataset_h{horizon}`` as columns.
    """
    agg = results.groupby(["model_name", "dataset_name", "horizon"])[metric].mean().reset_index()
    agg["config"] = agg["dataset_name"] + "_h" + agg["horizon"].astype(str)
    pivot = agg.pivot(index="model_name", columns="config", values=metric)
    # Sort rows by mean rank
    ranks = _mean_rank(results, metric)
    pivot = pivot.loc[pivot.index.intersection(ranks.sort_values().index)]
    return pivot


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def generate_heatmaps(
    results: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate accuracy heatmaps for long-term and M4 tracks.

    Args:
        results: Raw results DataFrame.
        output_dir: Directory where figures are written.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for task, metric, title in [
        ("long_term", "mae", "MAE — Long-term Forecast"),
        ("long_term", "mse", "MSE — Long-term Forecast"),
        ("m4", "owa", "OWA — M4 Short-term Forecast"),
    ]:
        subset = results[results["task"] == task]
        if subset.empty:
            continue

        pivot = _pivot_metrics(subset, metric)
        if pivot.empty:
            continue

        fig_h = max(6, len(pivot) * 0.4)
        fig_w = max(10, len(pivot.columns) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn_r",
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7},
            linewidths=0.4,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Dataset × Horizon", fontsize=9)
        ax.set_ylabel("Model", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()

        stem = f"heatmap_{task}_{metric}"
        for ext in ("png", "pdf"):
            path = output_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
            generated.append(path)
        plt.close(fig)

    return generated


# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

def generate_ranking_tables(
    results: pd.DataFrame,
    tables_dir: Path,
) -> list[Path]:
    """Generate CSV and LaTeX ranking tables for both benchmark tracks.

    Args:
        results: Raw results DataFrame.
        tables_dir: Output directory (``paper/tables/``).

    Returns:
        List of generated file paths.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # ---- Long-term ranking --------------------------------------------------
    lt = results[results["task"] == "long_term"]
    if not lt.empty:
        rank_mae = _mean_rank(lt, "mae").rename("rank_mae")
        rank_mse = _mean_rank(lt, "mse").rename("rank_mse")
        mean_mae = lt.groupby("model_name")["mae"].mean().rename("mean_mae")
        mean_mse = lt.groupby("model_name")["mse"].mean().rename("mean_mse")
        mean_rmse = lt.groupby("model_name")["rmse"].mean().rename("mean_rmse")

        lt_table = pd.concat([rank_mae, rank_mse, mean_mae, mean_mse, mean_rmse], axis=1)
        lt_table["avg_rank"] = (lt_table["rank_mae"] + lt_table["rank_mse"]) / 2
        lt_table = lt_table.sort_values("avg_rank").round(4)
        lt_table.index.name = "Model"

        csv_path = tables_dir / "long_term.csv"
        lt_table.to_csv(csv_path)
        generated.append(csv_path)

        tex_path = tables_dir / "long_term.tex"
        tex_path.write_text(_to_latex(lt_table, caption="Long-term forecast ranking."))
        generated.append(tex_path)

    # ---- M4 ranking ---------------------------------------------------------
    m4 = results[results["task"] == "m4"]
    if not m4.empty:
        rank_owa = _mean_rank(m4, "owa").rename("rank_owa")
        mean_owa = m4.groupby("model_name")["owa"].mean().rename("mean_owa")
        mean_smape = m4.groupby("model_name")["smape"].mean().rename("mean_smape")
        mean_mase = m4.groupby("model_name")["mase"].mean().rename("mean_mase")

        m4_table = pd.concat([rank_owa, mean_owa, mean_smape, mean_mase], axis=1)
        m4_table = m4_table.sort_values("rank_owa").round(4)
        m4_table.index.name = "Model"

        csv_path = tables_dir / "short_term.csv"
        m4_table.to_csv(csv_path)
        generated.append(csv_path)

        tex_path = tables_dir / "short_term.tex"
        tex_path.write_text(_to_latex(m4_table, caption="M4 short-term forecast ranking."))
        generated.append(tex_path)

    # ---- Efficiency leaderboard ---------------------------------------------
    if not results.empty:
        eff = _efficiency_leaderboard(results)
        if not eff.empty:
            csv_path = tables_dir / "efficiency.csv"
            eff.to_csv(csv_path)
            generated.append(csv_path)

            tex_path = tables_dir / "efficiency.tex"
            tex_path.write_text(_to_latex(eff, caption="Model efficiency leaderboard."))
            generated.append(tex_path)

    return generated


def _efficiency_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """Compute the composite efficiency leaderboard.

    Composite efficiency score = accuracy_rank / normalised_compute_cost,
    where compute_cost = 0.5 × normalised_train_time + 0.5 × normalised_latency.
    Higher score = better accuracy-per-compute-dollar.

    Args:
        results: Raw results DataFrame.

    Returns:
        DataFrame sorted by efficiency_score descending.
    """
    agg = results.groupby("model_name").agg(
        mean_mae=("mae", "mean"),
        mean_train_time_s=("train_time_s", "mean"),
        mean_latency_ms=("inference_latency_ms", "mean"),
        mean_peak_gpu_mb=("peak_gpu_mb", "mean"),
        mean_param_count=("param_count", "mean"),
    ).round(4)

    def _norm(col: pd.Series) -> pd.Series:
        mn, mx = col.min(), col.max()
        if mx == mn:
            return pd.Series(0.5, index=col.index)
        return (col - mn) / (mx - mn)

    agg["accuracy_rank"] = _mean_rank(results, "mae")
    norm_time = _norm(agg["mean_train_time_s"])
    norm_latency = _norm(agg["mean_latency_ms"])
    compute_cost = 0.5 * norm_time + 0.5 * norm_latency + 1e-8  # avoid div/0

    # Higher accuracy_rank means worse accuracy; invert for score
    n_models = len(agg)
    agg["efficiency_score"] = (n_models - agg["accuracy_rank"] + 1) / (n_models * compute_cost)
    agg = agg.sort_values("efficiency_score", ascending=False).round(4)
    agg.index.name = "Model"
    return agg


def _to_latex(df: pd.DataFrame, caption: str) -> str:
    """Convert a DataFrame to a LaTeX booktabs-style table string.

    Args:
        df: Table DataFrame.
        caption: LaTeX caption string.

    Returns:
        Full LaTeX table source.
    """
    col_str = " ".join(["l"] + ["r"] * len(df.columns))
    header = " & ".join(
        ["\\textbf{Model}"] + [f"\\textbf{{{c}}}" for c in df.columns]
    ) + " \\\\"

    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        rows.append(" & ".join(cells) + " \\\\")

    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\begin{{tabular}}{{{col_str}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
