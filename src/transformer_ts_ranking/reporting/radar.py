"""Radar (spider) charts comparing model accuracy and efficiency.

Each radar chart shows multiple normalised metrics as polygon areas, with one
polygon per model or model family.  Five axes are plotted:
  1. Accuracy     — 1 − normalised MAE   (higher = better)
  2. Efficiency   — 1 − normalised training time  (higher = better)
  3. Speed        — 1 − normalised inference latency  (higher = better)
  4. Memory       — 1 − normalised peak GPU MB  (higher = better)
  5. Stability    — 1 − normalised std of ranks across datasets  (higher = better)

All metrics are min-max normalised across the candidate models so all axes
are in [0, 1] and higher is always better.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_RADAR_AXES = [
    ("accuracy", "Accuracy\n(1−MAE_norm)"),
    ("efficiency", "Efficiency\n(1−time_norm)"),
    ("speed", "Speed\n(1−latency_norm)"),
    ("memory", "Memory\n(1−gpu_norm)"),
    ("stability", "Stability\n(1−rank_std_norm)"),
]


def _minmax_norm(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Min-max normalise; if ``lower_is_better``, invert so 1 = best."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    normed = (series - mn) / (mx - mn)
    return 1.0 - normed if lower_is_better else normed


def _build_radar_df(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run results into per-model radar coordinates.

    Args:
        results: Raw results DataFrame from parquet.

    Returns:
        DataFrame indexed by model_name with five normalised axis columns.
    """
    # Compute per-model mean metrics across all datasets/horizons/seeds
    agg = results.groupby("model_name").agg(
        mean_mae=("mae", "mean"),
        mean_train_time=("train_time_s", "mean"),
        mean_latency=("inference_latency_ms", "mean"),
        mean_gpu=("peak_gpu_mb", "mean"),
    )

    # Stability: std of per-dataset MAE rank (lower = more consistent)
    rank_stds: dict[str, float] = {}
    for model_name in agg.index:
        model_df = results[results["model_name"] == model_name]
        # Rank within each (dataset, horizon) group
        by_config = model_df.groupby(["dataset_name", "horizon"])["mae"].mean()
        # Compare against all other models for the same configs
        all_ranks = []
        for (ds, h), val in by_config.items():
            config_df = results[
                (results["dataset_name"] == ds) & (results["horizon"] == h)
            ].groupby("model_name")["mae"].mean()
            rank = (config_df < val).sum() + 1  # 1 = best
            all_ranks.append(rank)
        rank_stds[model_name] = float(np.std(all_ranks)) if all_ranks else float("nan")

    agg["rank_std"] = pd.Series(rank_stds)

    radar = pd.DataFrame(index=agg.index)
    radar["accuracy"] = _minmax_norm(agg["mean_mae"], lower_is_better=True)
    radar["efficiency"] = _minmax_norm(agg["mean_train_time"], lower_is_better=True)
    radar["speed"] = _minmax_norm(agg["mean_latency"], lower_is_better=True)
    radar["memory"] = _minmax_norm(agg["mean_gpu"], lower_is_better=True)
    radar["stability"] = _minmax_norm(agg["rank_std"], lower_is_better=True)
    return radar.fillna(0.0)


def _radar_plot(
    radar_df: pd.DataFrame,
    title: str,
    output_dir: Path,
    filename_stem: str,
    max_models: int = 8,
) -> list[Path]:
    """Generate a spider chart for the given radar DataFrame.

    Args:
        radar_df: Per-model normalised radar coordinates.
        title: Figure title.
        output_dir: Output directory.
        filename_stem: Base file name (without extension).
        max_models: Maximum models to overlay on one chart.

    Returns:
        List of generated file paths (PNG and PDF).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    axes_labels = [label for _, label in _RADAR_AXES]
    n_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    models = radar_df.index.tolist()[:max_models]
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis labels
    ax.set_thetagrids(np.degrees(angles[:-1]), axes_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(True, linewidth=0.5, alpha=0.6)

    for i, model_name in enumerate(models):
        row = radar_df.loc[model_name]
        values = [row[col] for col, _ in _RADAR_AXES]
        values += values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles, values, color=color, linewidth=1.8, label=model_name)
        ax.fill(angles, values, color=color, alpha=0.07)

    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8,
        framealpha=0.8,
    )

    fig.tight_layout()
    generated = []
    for ext in ("png", "pdf"):
        path = output_dir / f"{filename_stem}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        generated.append(path)
    plt.close(fig)
    return generated


def generate_radar_charts(
    results: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate radar charts from benchmark results.

    Produces three variants:
      1. All models — long-term track.
      2. Top-10 models by average MAE — long-term track.
      3. All models — M4 track.

    Args:
        results: Raw results DataFrame loaded from parquet.
        output_dir: Directory where figures are written.

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []

    lt_df = results[results["task"] == "long_term"].copy()
    if not lt_df.empty:
        radar_lt = _build_radar_df(lt_df)
        generated += _radar_plot(
            radar_lt,
            title="Long-term Forecast — Model Comparison",
            output_dir=output_dir,
            filename_stem="radar_long_term_all",
            max_models=21,
        )
        # Top-10 by MAE
        top10 = (
            lt_df.groupby("model_name")["mae"].mean().nsmallest(10).index.tolist()
        )
        if len(top10) >= 2:
            radar_top10 = radar_lt.loc[[m for m in top10 if m in radar_lt.index]]
            generated += _radar_plot(
                radar_top10,
                title="Long-term Forecast — Top-10 Models",
                output_dir=output_dir,
                filename_stem="radar_long_term_top10",
                max_models=10,
            )

    m4_df = results[results["task"] == "m4"].copy()
    if not m4_df.empty:
        # Use OWA instead of MAE for M4 radar accuracy axis
        m4_df = m4_df.copy()
        m4_df["mae"] = m4_df["owa"].fillna(m4_df["mae"])
        radar_m4 = _build_radar_df(m4_df)
        generated += _radar_plot(
            radar_m4,
            title="M4 Short-term Forecast — Model Comparison",
            output_dir=output_dir,
            filename_stem="radar_m4_all",
            max_models=21,
        )

    return generated
