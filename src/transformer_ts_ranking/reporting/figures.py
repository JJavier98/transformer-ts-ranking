"""Paper figures built from the scoped, seed-averaged results.

All figures are written as PNG (200 dpi) and PDF to ``paper/figures/``:

* ``rank_distribution_<metric>``  — boxplot of each model's rank over the
  core blocks (consistency vs. specialisation), ordered by mean rank.
* ``heatmap_rank_<metric>``       — model × block rank matrix.
* ``heatmap_relative_mae``        — MAE relative to the best model of each
  block (1.0 = best), so datasets of different scales share one colour map.
* ``horizon_scaling``             — MAE vs. horizon per dataset for the top
  models.
* ``m4_owa``                      — OWA per frequency and overall.
* ``convergence_<dataset>``       — mean validation loss per epoch.
* ``accuracy_vs_params`` / ``accuracy_vs_memory`` — mean rank against
  hardware-independent footprint.  Time-based efficiency figures come from
  the fixed-hardware profiling pass (past runs mixed GPU types).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .loading import ReportData  # noqa: E402
from .tables import block_ranks, m4_table, ranking_table  # noqa: E402


def _plain(label: str) -> str:
    """Display label without LaTeX markers (for matplotlib)."""
    return label.replace("$^\\dagger$", "†").replace("$", "")


def _save(fig: plt.Figure, stem: Path, written: list[Path]) -> None:
    """Save a figure as PNG and PDF and close it."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)


def _block_order(data: ReportData) -> list[str]:
    """Core blocks in protocol order (dataset, then horizon)."""
    return [f"{d}_h{h}" for d in data.scope["long_term"]["core_datasets"]
            for h in data.expected_horizons(d)]


def rank_distribution(data: ReportData, order: list[str], metric: str, out: Path,
                      written: list[Path]) -> None:
    """Boxplot of per-block ranks per model."""
    r = block_ranks(data.lt_core, metric)
    r["label"] = r["model_name"].map(lambda m: _plain(data.display(m)))
    labels = [_plain(data.display(m)) for m in order]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(order) + 1.2))
    sns.boxplot(data=r, y="label", x="rank", order=labels, ax=ax, color="#9ecae1",
                fliersize=2, linewidth=0.8)
    means = r.groupby("label")["rank"].mean().reindex(labels)
    ax.scatter(means.values, range(len(labels)), color="black", s=10, zorder=3, label="mean")
    ax.set_xlabel(f"Rank per dataset–horizon block ({metric.upper()}, 1 = best)")
    ax.set_ylabel("")
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.grid(axis="x", linewidth=0.3, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out / f"rank_distribution_{metric}", written)


def rank_heatmap(data: ReportData, order: list[str], metric: str, out: Path,
                 written: list[Path]) -> None:
    """Model × block rank heatmap."""
    r = block_ranks(data.lt_core, metric)
    r["block"] = r["dataset_name"] + "_h" + r["horizon"].astype(str)
    pivot = r.pivot(index="model_name", columns="block", values="rank")
    pivot = pivot.reindex(index=order, columns=[b for b in _block_order(data) if b in pivot])
    fig, ax = plt.subplots(figsize=(0.42 * pivot.shape[1] + 2.5, 0.3 * pivot.shape[0] + 1.5))
    sns.heatmap(pivot, ax=ax, cmap="viridis_r", annot=True, fmt=".0f", annot_kws={"size": 6},
                cbar_kws={"label": "rank (1 = best)", "shrink": 0.5}, linewidths=0.2)
    ax.set_yticklabels([_plain(data.display(m)) for m in pivot.index], fontsize=7)
    ax.set_xticklabels([c.replace("_h", "\n") for c in pivot.columns], fontsize=6, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, out / f"heatmap_rank_{metric}", written)


def relative_heatmap(data: ReportData, order: list[str], out: Path, written: list[Path]) -> None:
    """MAE divided by the best MAE of each block (log colour scale)."""
    cells = data.lt_core.copy()
    cells["block"] = cells["dataset_name"] + "_h" + cells["horizon"].astype(str)
    cells["rel"] = cells["mae"] / cells.groupby("block")["mae"].transform("min")
    pivot = cells.pivot(index="model_name", columns="block", values="rel")
    pivot = pivot.reindex(index=order, columns=[b for b in _block_order(data) if b in pivot])
    fig, ax = plt.subplots(figsize=(0.42 * pivot.shape[1] + 2.5, 0.3 * pivot.shape[0] + 1.5))
    sns.heatmap(np.log10(pivot), ax=ax, cmap="RdYlGn_r", annot=pivot.round(2), fmt="",
                annot_kws={"size": 5}, vmin=0, vmax=float(np.nanquantile(np.log10(pivot), 0.95)),
                cbar_kws={"label": "log10(MAE / best MAE)", "shrink": 0.5}, linewidths=0.2)
    ax.set_yticklabels([_plain(data.display(m)) for m in pivot.index], fontsize=7)
    ax.set_xticklabels([c.replace("_h", "\n") for c in pivot.columns], fontsize=6, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, out / "heatmap_relative_mae", written)


def horizon_scaling(data: ReportData, order: list[str], out: Path, written: list[Path],
                    n_models: int = 8) -> None:
    """MAE vs. horizon per dataset for the best-ranked models."""
    core = data.scope["long_term"]["core_datasets"]
    top = order[:n_models]
    ncols = 4
    nrows = int(np.ceil(len(core) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    palette = sns.color_palette("tab10", n_colors=len(top))
    for i, d in enumerate(core):
        ax = axes[i // ncols][i % ncols]
        sub = data.lt_core[data.lt_core["dataset_name"] == d]
        for color, m in zip(palette, top):
            s = sub[sub["model_name"] == m].sort_values("horizon")
            ax.plot(s["horizon"], s["mae"], marker="o", ms=3, lw=1.2, color=color,
                    label=_plain(data.display(m)))
        ax.set_title(data.display_dataset(d), fontsize=9)
        ax.set_xticks(data.expected_horizons(d))
        ax.tick_params(labelsize=7)
        ax.grid(linewidth=0.3, alpha=0.5)
    for j in range(len(core), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=2,
               bbox_to_anchor=(0.98, 0.05))
    fig.supxlabel("Horizon", fontsize=9)
    fig.supylabel("MAE (original scale)", fontsize=9)
    fig.tight_layout()
    _save(fig, out / "horizon_scaling", written)


def m4_figure(data: ReportData, out: Path, written: list[Path]) -> None:
    """OWA per frequency (heatmap) and overall (bars)."""
    if data.m4_cells.empty:
        return
    table, _ = m4_table(data)
    freqs = data.scope["m4"]["frequencies"]
    table = table.set_index("model_name")
    owa = table[[f"owa_{f}" for f in freqs if f"owa_{f}" in table]]
    owa.columns = [c.replace("owa_", "") for c in owa.columns]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.3 * len(table) + 1.5),
                                   gridspec_kw={"width_ratios": [3, 1.2]})
    sns.heatmap(owa, ax=ax1, cmap="RdYlGn_r", annot=True, fmt=".2f", annot_kws={"size": 6},
                center=1.0, cbar_kws={"label": "OWA (1 = Naive2)", "shrink": 0.5})
    ax1.set_yticklabels([_plain(data.display(m)) for m in owa.index], fontsize=7)
    ax1.set_ylabel("")
    if "owa_overall" in table:
        ax2.barh(range(len(table)), table["owa_overall"], color="#6baed6")
        ax2.axvline(1.0, color="black", lw=0.8, ls="--")
        ax2.set_ylim(len(table) - 0.5, -0.5)
        ax2.set_yticks([])
        ax2.set_xlabel("Overall OWA")
    _save(fig, out / "m4_owa", written)


def convergence_figures(data: ReportData, raw_dir: Path, order: list[str], out: Path,
                        written: list[Path], horizon_index: int = 0) -> None:
    """Mean validation loss per epoch (one panel per model) for every core dataset.

    Only the epoch logs of the runs selected by the de-duplication (same
    shard, model, dataset, horizon and seed) are used.
    """
    rows = data.lt_rows[data.lt_rows["ok"] & data.lt_rows["model_name"].isin(order)]
    wanted = set(zip(rows["shard"], rows["model_name"], rows["dataset_name"],
                     rows["horizon"], rows["seed"]))
    records = []
    for shard in rows["shard"].unique():
        path = raw_dir / "long_term" / shard / "epoch_logs.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if (shard, rec["model_name"], rec["dataset_name"], rec["horizon"],
                    rec["seed"]) in wanted:
                records.append(rec)
    if not records:
        return
    logs = pd.DataFrame(records).drop_duplicates(
        ["model_name", "dataset_name", "horizon", "seed", "epoch"], keep="last")
    for d in data.scope["long_term"]["core_datasets"]:
        h = data.expected_horizons(d)[horizon_index]
        sub = logs[(logs["dataset_name"] == d) & (logs["horizon"] == h)]
        models = [m for m in order if m in set(sub["model_name"])]
        if not models:
            continue
        ncols = 6
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.7 * nrows), squeeze=False)
        for i, m in enumerate(models):
            ax = axes[i // ncols][i % ncols]
            s = sub[sub["model_name"] == m]
            for _, seed_df in s.groupby("seed"):
                seed_df = seed_df.sort_values("epoch")
                ax.plot(seed_df["epoch"], seed_df["val_loss"], lw=0.6, alpha=0.4, color="firebrick")
                ax.plot(seed_df["epoch"], seed_df["train_loss"], lw=0.6, alpha=0.4, color="steelblue")
            ax.plot(*zip(*s.groupby("epoch")["val_loss"].mean().items()), lw=1.3,
                    color="firebrick", label="val")
            ax.plot(*zip(*s.groupby("epoch")["train_loss"].mean().items()), lw=1.3,
                    color="steelblue", label="train")
            ax.set_title(_plain(data.display(m)), fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_yscale("log")
        for j in range(len(models), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        axes[0][0].legend(fontsize=5)
        fig.suptitle(f"Training / validation loss — {data.display_dataset(d)}, H={h}", fontsize=9)
        fig.tight_layout()
        _save(fig, out / f"convergence_{d}_h{h}", written)


def footprint_figures(data: ReportData, order: list[str], out: Path, written: list[Path]) -> None:
    """Mean rank vs. parameters and vs. peak memory (hardware-independent)."""
    core = data.scope["long_term"]["core_datasets"]
    n_blocks = sum(len(data.expected_horizons(d)) for d in core)
    rank = ranking_table(data.lt_core, n_blocks).set_index("model_name")
    rows = data.lt_rows[data.lt_rows["ok"] & data.lt_rows["dataset_name"].isin(core)]
    foot = rows.groupby("model_name").agg(params=("param_count", "median"),
                                          mem=("peak_gpu_mb", "median"))
    df = rank.join(foot, how="inner")
    zero_shot = set(data.scope.get("zero_shot_models", []))
    for col, label, stem in (("params", "Trainable parameters (median)", "accuracy_vs_params"),
                             ("mem", "Peak training memory, MB (median)", "accuracy_vs_memory")):
        sub = df[df[col] > 0] if col == "params" else df
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for m, r in sub.iterrows():
            ax.scatter(r[col], r["avg_rank"], s=22,
                       color="#de2d26" if m in zero_shot else "#3182bd", zorder=3)
            ax.annotate(_plain(data.display(m)), (r[col], r["avg_rank"]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_ylabel("Average rank (MAE/MSE, lower = better)")
        ax.grid(linewidth=0.3, alpha=0.5)
        _save(fig, out / stem, written)


def generate_figures(data: ReportData, raw_dir: Path, figures_dir: Path) -> list[Path]:
    """Write every paper figure.

    Args:
        data: Scoped report data.
        raw_dir: ``results/raw`` (for the epoch logs).
        figures_dir: Output directory.

    Returns:
        Written file paths.
    """
    written: list[Path] = []
    core = data.scope["long_term"]["core_datasets"]
    n_blocks = sum(len(data.expected_horizons(d)) for d in core)
    order = list(ranking_table(data.lt_core, n_blocks)["model_name"])
    for metric in ("mae", "mse"):
        rank_distribution(data, order, metric, figures_dir, written)
        rank_heatmap(data, order, metric, figures_dir, written)
    relative_heatmap(data, order, figures_dir, written)
    horizon_scaling(data, order, figures_dir, written)
    m4_figure(data, figures_dir, written)
    convergence_figures(data, raw_dir, order, figures_dir / "convergence", written)
    footprint_figures(data, order, figures_dir, written)
    return written
