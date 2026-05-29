from pathlib import Path

import numpy as np
import pytest

from transformer_ts_ranking.bootstrap import materialize_bootstrap_manifests
from transformer_ts_ranking.data.m4 import load_m4_dataset
from transformer_ts_ranking.evaluation.m4_metrics import evaluate_m4_dataset, mase, smape


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yearly_dataset(tmp_path: Path):
    manifest_paths = materialize_bootstrap_manifests(
        repo_root=_repo_root(),
        config_dir=tmp_path / "configs" / "benchmark",
        audit_output_dir=tmp_path / "artifacts" / "audit",
    )
    return load_m4_dataset(
        repo_root=_repo_root(),
        frequency_label="Yearly",
        manifest_path=manifest_paths["m4_datasets"],
    )


def test_smape_and_mase_are_zero_for_perfect_forecasts() -> None:
    actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    insample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    assert smape(actual, actual) == 0.0
    assert mase(insample, actual, actual, seasonality=1) == 0.0


def test_evaluate_m4_dataset_returns_owa_below_one_when_beating_naive2(tmp_path: Path) -> None:
    dataset = _load_yearly_dataset(tmp_path)
    selected_ids = dataset.series_ids[:5]
    predictions = {
        series_id: dataset.series[series_id].test_values.copy()
        for series_id in selected_ids
    }

    subset = type(dataset)(
        frequency_label=dataset.frequency_label,
        frequency_code=dataset.frequency_code,
        horizon=dataset.horizon,
        m4_root=dataset.m4_root,
        info_path=dataset.info_path,
        train_path=dataset.train_path,
        test_path=dataset.test_path,
        naive2_path=dataset.naive2_path,
        series={series_id: dataset.series[series_id] for series_id in selected_ids},
    )

    result = evaluate_m4_dataset(subset, predictions)

    assert result.frequency_label == "Yearly"
    assert result.horizon == 6
    assert result.seasonality == 1
    assert result.mean_smape == 0.0
    assert result.mean_mase == 0.0
    assert result.mean_owa == 0.0


def test_evaluate_m4_dataset_requires_predictions_for_all_series(tmp_path: Path) -> None:
    dataset = _load_yearly_dataset(tmp_path)
    with pytest.raises(KeyError):
        evaluate_m4_dataset(dataset, predictions={})