"""Convergence curve plots from per-epoch training logs.

Reads ``results/raw/epoch_logs.jsonl`` and generates one figure per
(model, dataset, horizon) combination showing train-loss and val-loss
as a function of epoch.  Also generates a summary grid of convergence
curves for the paper.

All figures are saved as both PNG (for preview) and PDF (for LaTeX).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_epoch_logs(jsonl_path: Path) -> pd.DataFrame:
    """Load per-epoch JSONL logs into a tidy DataFrame.

    Args:
        jsonl_path: Path to ``epoch_logs.jsonl``.

    Returns:
        DataFrame with columns: run_id, model_name, dataset_name, horizon,
        seed, epoch, train_loss, val_loss, epoch_time_s, peak_gpu_mb.
    """
    records = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                import json
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return frame


def plot_convergence_curves(
    logs: pd.DataFrame,
    output_dir: Path,
    max_models_per_figure: int = 6,
) -> list[Path]:
    """Generate convergence curve figures from epoch logs.

    One figure shows train-loss and val-loss vs. epoch for all seeds of one
    (model, dataset, horizon) combination.  Seed replicates are shown as
    semi-transparent thin lines; their mean is the bold foreground line.

    Args:
        logs: Per-epoch log DataFrame from ``load_epoch_logs()``.
        output_dir: Directory where figures are written.
        max_models_per_figure: Maximum models on a single grid figure.

    Returns:
        List of paths to the generated figure files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    required = {"model_name", "dataset_name", "horizon", "seed", "epoch",
                "train_loss", "val_loss"}
    if logs.empty or not required.issubset(logs.columns):
        return generated

    groups = logs.groupby(["model_name", "dataset_name", "horizon"])

    for (model_name, dataset_name, horizon), group_df in groups:
        fig, ax = plt.subplots(figsize=(6, 4))

        seeds = sorted(group_df["seed"].unique())
        palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

        train_means: list[list[float]] = []
        val_means: list[list[float]] = []

        for i, seed in enumerate(seeds):
            seed_df = group_df[group_df["seed"] == seed].sort_values("epoch")
            color = palette[i % len(palette)]
            ax.plot(
                seed_df["epoch"],
                seed_df["train_loss"],
                color=color,
                alpha=0.3,
                linewidth=0.8,
                linestyle="-",
            )
            if seed_df["val_loss"].notna().any():
                ax.plot(
                    seed_df["epoch"],
                    seed_df["val_loss"],
                    color=color,
                    alpha=0.3,
                    linewidth=0.8,
                    linestyle="--",
                )
            train_means.append(seed_df["train_loss"].tolist())
            val_means.append(seed_df["val_loss"].tolist())

        # Mean across seeds (use the shortest seed run to align epochs)
        if train_means:
            min_len = min(len(v) for v in train_means)
            epochs_axis = list(range(min_len))
            import numpy as np
            mean_train = pd.DataFrame(
                {i: v[:min_len] for i, v in enumerate(train_means)}
            ).mean(axis=1)
            ax.plot(epochs_axis, mean_train, color="black", linewidth=2.0,
                    linestyle="-", label="Train (mean)")

            non_nan_val = [v for v in val_means if any(x == x for x in v[:min_len])]
            if non_nan_val:
                mean_val = pd.DataFrame(
                    {i: v[:min_len] for i, v in enumerate(non_nan_val)}
                ).mean(axis=1)
                ax.plot(epochs_axis, mean_val, color="black", linewidth=2.0,
                        linestyle="--", label="Val (mean)")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{model_name} — {dataset_name} h={horizon}")
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()

        stem = f"convergence_{model_name}_{dataset_name}_h{horizon}"
        for ext in ("png", "pdf"):
            path = output_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
            generated.append(path)
        plt.close(fig)

    # ---- Summary grid: all models for one representative dataset/horizon ----
    representative = _pick_representative(logs)
    if representative:
        grid_path = _plot_convergence_grid(
            logs=logs,
            dataset_name=representative[0],
            horizon=representative[1],
            output_dir=output_dir,
            max_models=max_models_per_figure,
        )
        generated.extend(grid_path)

    return generated


def _pick_representative(logs: pd.DataFrame) -> tuple[str, int] | None:
    """Pick the (dataset, horizon) combination with the most model coverage."""
    if logs.empty:
        return None
    counts = logs.groupby(["dataset_name", "horizon"])["model_name"].nunique()
    if counts.empty:
        return None
    best_idx = counts.idxmax()
    return best_idx[0], best_idx[1]


def _plot_convergence_grid(
    logs: pd.DataFrame,
    dataset_name: str,
    horizon: int,
    output_dir: Path,
    max_models: int,
) -> list[Path]:
    """Generate a multi-panel convergence grid for one dataset/horizon.

    Args:
        logs: Full epoch log DataFrame.
        dataset_name: Dataset to highlight.
        horizon: Horizon to highlight.
        output_dir: Output directory.
        max_models: Maximum models to include.

    Returns:
        Paths to the generated grid figure.
    """
    subset = logs[(logs["dataset_name"] == dataset_name) & (logs["horizon"] == horizon)]
    if subset.empty:
        return []

    models = sorted(subset["model_name"].unique())[:max_models]
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"Convergence — {dataset_name}  h={horizon}", fontsize=12)

    for idx, model_name in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        model_df = subset[subset["model_name"] == model_name]

        for seed, seed_df in model_df.groupby("seed"):
            seed_df = seed_df.sort_values("epoch")
            ax.plot(seed_df["epoch"], seed_df["train_loss"],
                    alpha=0.4, linewidth=0.9, color="steelblue")
            if seed_df["val_loss"].notna().any():
                ax.plot(seed_df["epoch"], seed_df["val_loss"],
                        alpha=0.4, linewidth=0.9, color="salmon", linestyle="--")

        mean_train = model_df.groupby("epoch")["train_loss"].mean()
        ax.plot(mean_train.index, mean_train.values, color="navy", linewidth=1.8,
                label="Train")
        mean_val = model_df.groupby("epoch")["val_loss"].mean().dropna()
        if not mean_val.empty:
            ax.plot(mean_val.index, mean_val.values, color="firebrick",
                    linewidth=1.8, linestyle="--", label="Val")

        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    # Hide empty subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    generated = []
    for ext in ("png", "pdf"):
        path = output_dir / f"convergence_grid_{dataset_name}_h{horizon}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        generated.append(path)
    plt.close(fig)
    return generated
