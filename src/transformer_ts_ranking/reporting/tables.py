"""Paper tables (CSV + LaTeX) built from the scoped, seed-averaged results.

Tables written to ``paper/tables/`` (LaTeX, booktabs) and, as CSV, to
``results/leaderboards/``:

* ``long_term_main``      — models × core datasets, MSE/MAE averaged over the
                            four horizons; best in bold, second underlined.
* ``long_term_<dataset>`` — per-dataset detail (models × horizons), mean ±
                            std over seeds (appendix).
* ``long_term_full.csv``  — every cell: mean, std, number of seeds, seeds.
* ``ranking_long_term``   — mean rank (MAE, MSE), wins, top-3 counts.
* ``m4``                  — OWA / sMAPE / MASE per frequency and overall
                            (series-weighted, as in the M4 competition).
* ``coverage``            — protocol coverage per model and status.
* ``seed_variability``    — dispersion of MAE across seeds per model.
* ``supplementary_long_term`` — electricity / traffic (partial).
* ``model_footprint``     — parameters and peak training memory.  Timing is
                            NOT reported here: past runs mixed V100 and A100
                            nodes; it comes from the fixed-hardware profiling.

Metrics are averaged across seeds and, within one dataset, across horizons;
they are never averaged across datasets (different scales) — cross-dataset
summaries use ranks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .loading import ReportData, coverage_table


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Escape LaTeX special characters in plain text (keeps ``$...$`` math)."""
    if "$" in text:
        return text
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def _fmt(value: float, digits: int) -> str:
    """Format a number compactly for a table cell (scientific when large)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if abs(value) >= 10 ** 5 or (0 < abs(value) < 10 ** -3):
        return f"{value:.{digits - 1}e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _best_marks(column: pd.Series) -> dict[str, str]:
    """Bold for the lowest value of a column, underline for the second lowest."""
    order = column.dropna().sort_values()
    marks = {}
    if len(order) > 0:
        marks[order.index[0]] = "best"
    if len(order) > 1:
        marks[order.index[1]] = "second"
    return marks


def _decorate(text: str, mark: str | None) -> str:
    """Apply the bold/underline mark to a cell."""
    if mark == "best":
        return rf"\textbf{{{text}}}"
    if mark == "second":
        return rf"\underline{{{text}}}"
    return text


def _latex_table(
    header_rows: list[str],
    body_rows: list[str],
    col_spec: str,
    caption: str,
    label: str,
    notes: list[str] | None = None,
    resize: bool = True,
) -> str:
    """Assemble a booktabs table (optionally resized to the text width)."""
    inner = "\n".join([
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *header_rows,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    if resize:
        inner = rf"\resizebox{{\textwidth}}{{!}}{{%" + "\n" + inner + "\n}"
    parts = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", inner]
    if notes:
        parts.append(r"\par\vspace{2pt}\footnotesize " + " ".join(notes))
    parts.append(r"\end{table}")
    return "\n".join(parts) + "\n"


def _write(df: pd.DataFrame, csv_dir: Path, name: str, latex: str | None,
           tex_dir: Path, written: list[Path]) -> None:
    """Write a CSV (and optionally its LaTeX rendering)."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    csv = csv_dir / f"{name}.csv"
    df.to_csv(csv, index=False)
    written.append(csv)
    if latex is not None:
        tex = tex_dir / f"{name}.tex"
        tex.write_text(latex)
        written.append(tex)


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def block_ranks(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank models within every (dataset, horizon) block (1 = best)."""
    out = cells[["model_name", "dataset_name", "horizon", metric]].copy()
    out["rank"] = out.groupby(["dataset_name", "horizon"])[metric].rank(method="average")
    return out


def ranking_table(cells: pd.DataFrame, n_blocks_expected: int) -> pd.DataFrame:
    """Mean rank (MAE, MSE), wins and top-3 counts per model over core blocks.

    Args:
        cells: Seed-aggregated core cells.
        n_blocks_expected: Number of blocks in the full protocol.

    Returns:
        Ranking sorted by average of the MAE and MSE mean ranks.
    """
    rows = {}
    for metric in ("mae", "mse"):
        r = block_ranks(cells, metric)
        g = r.groupby("model_name")["rank"]
        rows[f"mean_rank_{metric}"] = g.mean()
        rows[f"wins_{metric}"] = g.apply(lambda s: int((s == 1).sum()))
        rows[f"top3_{metric}"] = g.apply(lambda s: int((s <= 3).sum()))
    table = pd.DataFrame(rows)
    table["blocks"] = cells.groupby("model_name").size()
    table["complete"] = table["blocks"] == n_blocks_expected
    table["avg_rank"] = (table["mean_rank_mae"] + table["mean_rank_mse"]) / 2
    table = table.sort_values(["complete", "avg_rank"], ascending=[False, True])
    table.insert(0, "position", range(1, len(table) + 1))
    return table.reset_index().rename(columns={"index": "model_name"})


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------

def m4_naive2_reference(m4_rows: pd.DataFrame) -> pd.DataFrame:
    """Recover the Naive2 sMAPE and MASE of every M4 frequency from the runs.

    Every run stores ``owa = 0.5 mase / MASE_N2 + 0.5 smape / sMAPE_N2``
    with the same Naive2 reference for all models of a frequency, so the two
    reference values follow from a least-squares fit over the models.  The
    fit residual is reported as a consistency check.

    Args:
        m4_rows: Successful de-duplicated M4 rows.

    Returns:
        One row per frequency: ``smape_naive2``, ``mase_naive2``,
        ``max_abs_residual``.
    """
    out = []
    for freq, sub in m4_rows.groupby("dataset_name"):
        a = sub[["mase", "smape"]].to_numpy(dtype=float)
        b = sub["owa"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        resid = np.abs(a @ coef - b).max()
        out.append({"frequency": freq, "mase_naive2": 0.5 / coef[0],
                    "smape_naive2": 0.5 / coef[1], "max_abs_residual": float(resid)})
    return pd.DataFrame(out)


def m4_table(data: ReportData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-frequency and overall M4 metrics per model.

    The overall sMAPE and MASE are averages over all 100,000 series, i.e.
    per-frequency values weighted by the number of series; the overall OWA
    divides them by the equally weighted Naive2 values.

    Args:
        data: Scoped report data.

    Returns:
        ``(table, naive2)``.
    """
    cells = data.m4_cells
    rows_ok = data.m4_rows[data.m4_rows["ok"]]
    naive2 = m4_naive2_reference(rows_ok)
    n_series = data.scope["m4"]["n_series"]
    freqs = data.scope["m4"]["frequencies"]

    wide = cells.pivot_table(index="model_name", columns="dataset_name",
                             values=["owa", "smape", "mase"])
    table = pd.DataFrame(index=wide.index)
    for f in freqs:
        if ("owa", f) in wide:
            table[f"owa_{f}"] = wide[("owa", f)]
    complete = wide["owa"][freqs].notna().all(axis=1) if set(freqs) <= set(wide["owa"].columns) \
        else pd.Series(False, index=wide.index)
    weights = pd.Series({f: n_series[f] for f in freqs}, dtype=float)
    n2 = naive2.set_index("frequency")
    if complete.any() and set(freqs) <= set(n2.index):
        smape = (wide["smape"][freqs] * weights).sum(axis=1) / weights.sum()
        mase = (wide["mase"][freqs] * weights).sum(axis=1) / weights.sum()
        smape_n2 = float((n2.loc[freqs, "smape_naive2"] * weights).sum() / weights.sum())
        mase_n2 = float((n2.loc[freqs, "mase_naive2"] * weights).sum() / weights.sum())
        table["smape_overall"] = smape.where(complete)
        table["mase_overall"] = mase.where(complete)
        table["owa_overall"] = (0.5 * mase / mase_n2 + 0.5 * smape / smape_n2).where(complete)
    ranks = cells.assign(rank=cells.groupby("dataset_name")["owa"].rank(method="average"))
    table["mean_rank_owa"] = ranks.groupby("model_name")["rank"].mean()
    table = table.sort_values(["owa_overall" if "owa_overall" in table else "mean_rank_owa"])
    return table.reset_index(), naive2


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_tables(data: ReportData, tables_dir: Path, csv_dir: Path) -> list[Path]:
    """Write every paper table.

    Args:
        data: Scoped report data.
        tables_dir: LaTeX output (``paper/tables``).
        csv_dir: CSV output (``results/leaderboards``).

    Returns:
        Written file paths.
    """
    written: list[Path] = []
    disp: Callable[[str], str] = lambda m: _escape(data.display(m))
    core = data.scope["long_term"]["core_datasets"]
    cells = data.lt_core
    n_blocks = sum(len(data.expected_horizons(d)) for d in core)
    zero_shot = set(data.scope.get("zero_shot_models", []))
    variants = set(data.scope.get("library_variants", {}))

    def name_cell(model: str) -> str:
        """Display name with markers for zero-shot models and library variants."""
        text = disp(model)
        if model in zero_shot:
            text += r"$^{\ast}$"
        return text

    marker_notes = [r"$^{\ast}$ zero-shot pretrained foundation model.",
                    r"$^\dagger$ S-TransformerTS library variant (not a published architecture)."]

    # ---- ranking ----------------------------------------------------------
    ranking = ranking_table(cells, n_blocks)
    order = list(ranking["model_name"])
    header = (r"\# & Model & \multicolumn{3}{c}{MAE} & \multicolumn{3}{c}{MSE} & Avg. rank \\"
              "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"
              "\n" r" & & Mean rank & Wins & Top-3 & Mean rank & Wins & Top-3 & \\")
    body = []
    for _, r in ranking.iterrows():
        flag = "" if r["complete"] else r"$^{\S}$"
        body.append(" & ".join([
            str(r["position"]), name_cell(r["model_name"]) + flag,
            f"{r['mean_rank_mae']:.2f}", str(r["wins_mae"]), str(r["top3_mae"]),
            f"{r['mean_rank_mse']:.2f}", str(r["wins_mse"]), str(r["top3_mse"]),
            f"{r['avg_rank']:.2f}",
        ]) + r" \\")
    notes = marker_notes + [rf"Ranks over {n_blocks} dataset--horizon blocks "
                            r"(7 datasets $\times$ 4 horizons), 1 = best.",
                            r"$^{\S}$ incomplete coverage: ranked over the available blocks only."]
    _write(ranking, csv_dir, "ranking_long_term",
           _latex_table([header], body, "rlrrrrrrr",
                        "Long-term forecasting: average rank of every model.",
                        "tab:ranking_long_term", notes), tables_dir, written)

    # ---- main table: models × datasets (avg over horizons) ------------------
    per_ds = cells.groupby(["model_name", "dataset_name"])[["mse", "mae"]].mean()
    n_h = cells.groupby(["model_name", "dataset_name"]).size()
    main = per_ds.unstack("dataset_name")
    main_csv = main.copy()
    main_csv.columns = [f"{m}_{d}" for m, d in main_csv.columns]
    main_csv = main_csv.reindex(order).reset_index()
    marks = {(m, d): _best_marks(main[(m, d)]) for m in ("mse", "mae") for d in core
             if (m, d) in main}
    header = ["Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{_escape(data.display_dataset(d))}}}"
                                      for d in core) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(core))),
              " & " + " & ".join(["MSE & MAE"] * len(core)) + r" \\"]
    body = []
    for model in order:
        cells_txt = []
        for d in core:
            for m in ("mse", "mae"):
                v = main.loc[model, (m, d)] if (m, d) in main and model in main.index else np.nan
                partial = model in n_h.index.get_level_values(0) and \
                    n_h.get((model, d), 0) < len(data.expected_horizons(d))
                txt = _fmt(v, 3) + (r"$^{\S}$" if partial and not np.isnan(v) else "")
                cells_txt.append(_decorate(txt, marks.get((m, d), {}).get(model)))
        body.append(name_cell(model) + " & " + " & ".join(cells_txt) + r" \\")
    notes = marker_notes + [
        "Errors in the original scale of each dataset, averaged over 3 seeds and over the "
        "4 horizons; best in bold, second underlined. Values are comparable within a column only.",
        r"$^{\S}$ average over the available horizons only."]
    _write(main_csv, csv_dir, "long_term_main",
           _latex_table(header, body, "l" + "rr" * len(core),
                        "Long-term forecasting results (average over horizons).",
                        "tab:long_term_main", notes), tables_dir, written)

    # ---- full cell table + per-dataset appendix tables -----------------------
    full = cells.copy()
    full["display"] = full["model_name"].map(data.display)
    _write(full, csv_dir, "long_term_full", None, tables_dir, written)
    for d in core:
        sub = cells[cells["dataset_name"] == d]
        horizons = data.expected_horizons(d)
        header = ["Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$H={h}$}}" for h in horizons)
                  + r" \\",
                  "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(horizons))),
                  " & " + " & ".join(["MSE & MAE"] * len(horizons)) + r" \\"]
        idx = sub.set_index(["model_name", "horizon"])
        col_marks = {(m, h): _best_marks(sub[sub["horizon"] == h].set_index("model_name")[m])
                     for m in ("mse", "mae") for h in horizons}
        body = []
        for model in order:
            parts = []
            for h in horizons:
                for m in ("mse", "mae"):
                    if (model, h) in idx.index:
                        mean = idx.loc[(model, h), m]
                        std = idx.loc[(model, h), f"{m}_std"]
                        txt = _fmt(mean, 3) + (rf"\tiny{{$\pm${_fmt(std, 2)}}}"
                                               if not np.isnan(std) else "")
                        parts.append(_decorate(txt, col_marks[(m, h)].get(model)))
                    else:
                        parts.append("--")
            body.append(name_cell(model) + " & " + " & ".join(parts) + r" \\")
        latex = _latex_table(header, body, "l" + "rr" * len(horizons),
                             f"{_escape(data.display_dataset(d))}: mean $\\pm$ std over seeds.",
                             f"tab:lt_{d}", marker_notes)
        (tables_dir / f"long_term_{d}.tex").write_text(latex)
        written.append(tables_dir / f"long_term_{d}.tex")

    # ---- M4 -----------------------------------------------------------------
    if not data.m4_cells.empty:
        m4, naive2 = m4_table(data)
        _write(naive2, csv_dir, "m4_naive2_reference", None, tables_dir, written)
        freqs = data.scope["m4"]["frequencies"]
        cols = [f"owa_{f}" for f in freqs] + ["smape_overall", "mase_overall", "owa_overall"]
        cols = [c for c in cols if c in m4]
        mk = {c: _best_marks(m4.set_index("model_name")[c]) for c in cols}
        header = ["Model & " + " & ".join(f.replace("_", " ") for f in freqs)
                  + r" & sMAPE & MASE & OWA \\",
                  rf"\cmidrule(lr){{2-{1 + len(freqs)}}}\cmidrule(lr){{{2 + len(freqs)}-{4 + len(freqs)}}}",
                  r" & \multicolumn{" + str(len(freqs)) + r"}{c}{OWA per frequency} & "
                  r"\multicolumn{3}{c}{Overall} \\"]
        body = []
        for _, r in m4.iterrows():
            body.append(name_cell(r["model_name"]) + " & " + " & ".join(
                _decorate(_fmt(r[c], 3), mk[c].get(r["model_name"])) for c in cols) + r" \\")
        notes = marker_notes + [
            "OWA relative to Naive2 (1.0 = Naive2; lower is better). Overall sMAPE/MASE average "
            "all 100,000 series, i.e. frequencies weighted by their number of series."]
        _write(m4, csv_dir, "m4", _latex_table(header, body, "l" + "r" * len(cols),
                                               "M4 short-term forecasting.", "tab:m4", notes),
               tables_dir, written)

    # ---- coverage -------------------------------------------------------------
    cov = coverage_table(data)
    header = [r"Model & Core cells & Core runs & Diverged runs & Replacement-seed runs & Suppl. cells & M4 & Status \\"]
    body = []
    for _, r in cov.iterrows():
        body.append(" & ".join([_escape(r["display"]), r["core_cells"], str(r["core_runs"]),
                                str(r["core_diverged_runs"]),
                                str(r["core_replacement_seed_runs"]), r["supplementary_cells"],
                                r["m4_cells"], _escape(r["status"].replace("_", " "))]) + r" \\")
    notes = [_escape(f"{m}: {v['reason']}") for m, v in data.excluded.items()]
    _write(cov, csv_dir, "coverage",
           _latex_table(header, body, "lrrrrrrl", "Benchmark coverage per model.",
                        "tab:coverage", notes), tables_dir, written)

    # ---- seed variability -----------------------------------------------------
    cv = cells.assign(cv_mae=cells["mae_std"] / cells["mae"])
    seedvar = cv.groupby("model_name").agg(
        median_cv_mae=("cv_mae", "median"), max_cv_mae=("cv_mae", "max"),
        cells_single_seed=("n_seeds", lambda s: int((s < 2).sum())),
    ).reindex(order).reset_index()
    body = [f"{name_cell(r['model_name'])} & {_fmt(r['median_cv_mae'] * 100, 2)} & "
            f"{_fmt(r['max_cv_mae'] * 100, 2)} \\\\" for _, r in seedvar.iterrows()]
    _write(seedvar, csv_dir, "seed_variability",
           _latex_table([r"Model & Median CV(MAE) \% & Max CV(MAE) \% \\"], body, "lrr",
                        "Sensitivity to the random seed: coefficient of variation of the MAE "
                        "across seeds, over the core cells.", "tab:seed_variability",
                        marker_notes, resize=False), tables_dir, written)

    # ---- supplementary --------------------------------------------------------
    if not data.lt_supp.empty:
        supp = data.lt_supp.copy()
        supp["display"] = supp["model_name"].map(data.display)
        _write(supp, csv_dir, "supplementary_long_term", None, tables_dir, written)
        datasets = data.scope["long_term"]["supplementary_datasets"]
        wide = supp.pivot_table(index="model_name", columns=["dataset_name", "horizon"],
                                values="mae")
        header = ["Model & " + " & ".join(
            rf"\multicolumn{{4}}{{c}}{{{_escape(data.display_dataset(d))} (MAE)}}" for d in datasets)
            + r" \\",
            " & " + " & ".join(f"{h}" for d in datasets for h in data.expected_horizons(d)) + r" \\"]
        body = []
        for model in [m for m in order if m in wide.index] + \
                [m for m in wide.index if m not in order]:
            vals = [_fmt(wide.loc[model].get((d, h), np.nan), 1)
                    for d in datasets for h in data.expected_horizons(d)]
            body.append(name_cell(model) + " & " + " & ".join(vals) + r" \\")
        _write(wide.reset_index(), csv_dir, "supplementary_long_term_wide",
               _latex_table(header, body, "l" + "r" * 8,
                            "Supplementary: high-dimensional datasets (partial coverage; "
                            "-- = not completed within the compute budget).",
                            "tab:supplementary", marker_notes), tables_dir, written)

    # ---- model footprint (hardware-independent) -----------------------------
    lt_ok = data.lt_rows[data.lt_rows["ok"] & data.lt_rows["model_name"].isin(order)
                         & data.lt_rows["dataset_name"].isin(core)]
    foot = lt_ok.groupby("model_name").agg(
        params=("param_count", "median"), peak_gpu_mb_median=("peak_gpu_mb", "median"),
        peak_gpu_mb_max=("peak_gpu_mb", "max"), epochs_median=("epochs_trained", "median"),
    ).reindex(order).reset_index()
    body = [f"{name_cell(r['model_name'])} & {_fmt(r['params'] / 1e6, 2)} & "
            f"{_fmt(r['peak_gpu_mb_median'] / 1024, 2)} & {_fmt(r['peak_gpu_mb_max'] / 1024, 2)} & "
            f"{_fmt(r['epochs_median'], 0)} \\\\" for _, r in foot.iterrows()]
    _write(foot, csv_dir, "model_footprint",
           _latex_table([r"Model & Params (M) & Peak mem. median (GB) & Peak mem. max (GB) & "
                         r"Epochs (median) \\"], body, "lrrrr",
                        "Model footprint over the core long-term runs (batch size 16).",
                        "tab:footprint", marker_notes + [
                            "Parameter counts depend on the number of channels; medians over "
                            "all core runs. Zero-shot models report 0 trainable parameters."],
                        resize=False), tables_dir, written)
    return written
