"""Tests for the paper reporting pipeline (synthetic shards, no models)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from transformer_ts_ranking.reporting.loading import aggregate_seeds, build_report_data, deduplicate
from transformer_ts_ranking.reporting.runner import ReportRunner, ReportRunnerConfig
from transformer_ts_ranking.reporting.statistics import critical_difference, friedman_test
from transformer_ts_ranking.reporting.tables import m4_naive2_reference

SCOPE = {
    "version": "test",
    "long_term": {"core_datasets": ["A", "B"], "supplementary_datasets": ["C"],
                  "horizons": {"default": [96, 192]}},
    "m4": {"frequencies": ["Yearly", "Hourly"], "n_series": {"Yearly": 100, "Hourly": 10}},
    "seeds": {"protocol": [42, 123], "replacement_order": [7]},
    "excluded_models": {"slow": {"status": "not_measured", "reason": "too slow"}},
    "display_names": {"m0": "Model-0"},
}


def _row(model, dataset, horizon, seed, mae, task="long_term", error=None, ts="2026-01-01T00:00:00+00:00",
         **extra):
    """One synthetic result row."""
    row = {"model_name": model, "dataset_name": dataset, "horizon": horizon, "seed": seed,
           "task": task, "mae": mae, "mse": mae ** 2, "rmse": mae, "smape": np.nan,
           "mase": np.nan, "owa": np.nan, "error": error, "run_timestamp": ts,
           "precision": "fp32", "param_count": 1000, "peak_gpu_mb": 100.0,
           "epochs_trained": 5, "gpu_name": "GPU", "train_time_s": 1.0,
           "inference_latency_ms": 1.0}
    row.update(extra)
    return row


def test_deduplicate_prefers_latest_successful_row():
    """A later error row must not replace an earlier successful run."""
    rows = pd.DataFrame([
        _row("m0", "A", 96, 42, 1.0, ts="2026-01-01T00:00:00+00:00"),
        _row("m0", "A", 96, 42, np.nan, error="OOM", ts="2026-01-02T00:00:00+00:00"),
        _row("m1", "A", 96, 42, np.nan, error="OOM", ts="2026-01-01T00:00:00+00:00"),
        _row("m1", "A", 96, 42, 2.0, ts="2026-01-03T00:00:00+00:00"),
    ])
    out = deduplicate(rows).set_index("model_name")
    assert out.loc["m0", "ok"] and out.loc["m0", "mae"] == 1.0
    assert out.loc["m1", "ok"] and out.loc["m1", "mae"] == 2.0


def test_aggregate_seeds_mean_std_and_seed_list():
    """Seeds are averaged per cell; the seed list is recorded."""
    rows = pd.DataFrame([_row("m0", "A", 96, s, v) for s, v in ((42, 1.0), (123, 3.0), (7, 2.0))])
    cell = aggregate_seeds(rows, ["mae"]).iloc[0]
    assert cell["mae"] == pytest.approx(2.0)
    assert cell["mae_std"] == pytest.approx(1.0)
    assert cell["n_seeds"] == 3 and cell["seeds"] == "7,42,123"


@pytest.mark.parametrize("k, q", [(2, 1.960), (5, 2.728), (10, 3.164)])
def test_critical_difference_matches_demsar_table(k, q):
    """CD uses the studentised range exactly as Demšar (2006), Table 5."""
    n = 14
    expected = q * np.sqrt(k * (k + 1) / (6 * n))
    assert critical_difference(k, n) == pytest.approx(expected, rel=2e-3)


def test_friedman_detects_consistent_ordering():
    """A model that is always best yields a significant test and W = 1."""
    values = pd.DataFrame({f"b{i}": [1.0, 2.0, 3.0, 4.0] for i in range(10)},
                          index=["m0", "m1", "m2", "m3"])
    res = friedman_test(values)
    assert res["significant_at_0_05"]
    assert res["kendall_W"] == pytest.approx(1.0)
    assert list(res["mean_ranks"]) == ["m0", "m1", "m2", "m3"]


def test_m4_naive2_reference_is_recovered():
    """Naive2 sMAPE/MASE are recovered from the stored OWA of several models."""
    smape_n2, mase_n2 = 16.0, 4.0
    rng = np.random.default_rng(0)
    rows = []
    for m in range(5):
        smape, mase = rng.uniform(10, 20), rng.uniform(2, 5)
        rows.append({"dataset_name": "Yearly", "smape": smape, "mase": mase,
                     "owa": 0.5 * mase / mase_n2 + 0.5 * smape / smape_n2})
    ref = m4_naive2_reference(pd.DataFrame(rows)).iloc[0]
    assert ref["smape_naive2"] == pytest.approx(smape_n2)
    assert ref["mase_naive2"] == pytest.approx(mase_n2)


def _write_shards(raw: Path) -> None:
    """Write a small synthetic set of long-term and M4 shards."""
    rng = np.random.default_rng(1)
    lt = []
    for i, model in enumerate(["m0", "m1", "m2", "slow"]):
        for d in ("A", "B", "C"):
            for h in (96, 192):
                for s in (42, 123):
                    lt.append(_row(model, d, h, s, 1.0 + i + rng.uniform(0, 0.1)))
    (raw / "long_term" / "shard1").mkdir(parents=True)
    pd.DataFrame(lt).to_parquet(raw / "long_term" / "shard1" / "results_raw.parquet")
    (raw / "long_term" / "shard1" / "epoch_logs.jsonl").write_text("\n".join(
        json.dumps({"model_name": r["model_name"], "dataset_name": r["dataset_name"],
                    "horizon": r["horizon"], "seed": r["seed"], "epoch": e,
                    "train_loss": 1.0 / (e + 1), "val_loss": 1.2 / (e + 1)})
        for r in lt for e in range(3)))
    m4 = []
    for i, model in enumerate(["m0", "m1", "m2"]):
        for f in ("Yearly", "Hourly"):
            smape, mase = 10 + i + rng.uniform(), 2 + i + rng.uniform()
            for s in (42, 123):
                m4.append(_row(model, f, 6, s, np.nan, task="m4", smape=smape, mase=mase,
                               owa=0.5 * mase / 3.0 + 0.5 * smape / 12.0))
    (raw / "m4" / "Y").mkdir(parents=True)
    pd.DataFrame(m4).to_parquet(raw / "m4" / "Y" / "results_raw.parquet")


def test_report_pipeline_end_to_end(tmp_path):
    """The runner writes tables, statistics, figures and a manifest."""
    _write_shards(tmp_path / "results" / "raw")
    scope_path = tmp_path / "scope.yaml"
    scope_path.write_text(yaml.safe_dump(SCOPE))

    data = build_report_data(tmp_path / "results" / "raw", scope_path)
    assert "slow" not in data.ranked_models
    assert set(data.lt_core["dataset_name"]) == {"A", "B"}

    manifest = ReportRunner(ReportRunnerConfig(repo_root=tmp_path, scope_path=scope_path)).run()
    stats = manifest["statistics"]["long_term_mae"]
    assert stats["n_models"] == 3 and stats["n_blocks"] == 4
    ranking = pd.read_csv(tmp_path / "results" / "leaderboards" / "ranking_long_term.csv")
    assert list(ranking["model_name"]) == ["m0", "m1", "m2"]
    assert (tmp_path / "paper" / "tables" / "long_term_main.tex").exists()
    assert (tmp_path / "paper" / "figures" / "cd_long_term_mae.pdf").exists()
    m4 = pd.read_csv(tmp_path / "results" / "leaderboards" / "m4.csv")
    assert m4["owa_overall"].notna().all()
    assert (tmp_path / "results" / "report_manifest.json").exists()
