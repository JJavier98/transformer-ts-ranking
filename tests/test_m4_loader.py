from pathlib import Path

import numpy as np
import pytest

from transformer_ts_ranking.bootstrap import materialize_bootstrap_manifests
from transformer_ts_ranking.data.m4 import load_m4_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _materialize_m4_manifest(tmp_path: Path) -> Path:
    manifest_paths = materialize_bootstrap_manifests(
        repo_root=_repo_root(),
        config_dir=tmp_path / "configs" / "benchmark",
        audit_output_dir=tmp_path / "artifacts" / "audit",
    )
    return manifest_paths["m4_datasets"]


def test_load_m4_dataset_uses_official_metadata_and_naive2(tmp_path: Path) -> None:
    dataset = load_m4_dataset(
        repo_root=_repo_root(),
        frequency_label="Yearly",
        manifest_path=_materialize_m4_manifest(tmp_path),
    )

    assert dataset.frequency_label == "Yearly"
    assert dataset.frequency_code == 1
    assert dataset.horizon == 6
    assert dataset.series_count > 0

    first_series = dataset.series["Y1"]
    assert first_series.category == "Macro"
    assert len(first_series.train_values) == 31
    assert len(first_series.test_values) == 6
    assert len(first_series.naive2_forecast) == 6
    assert np.isclose(first_series.train_values[-1], 7261.1)
    assert np.allclose(first_series.naive2_forecast, np.repeat(7261.1, 6))


def test_load_m4_dataset_rejects_unknown_frequency(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        load_m4_dataset(
            repo_root=_repo_root(),
            frequency_label="Unknown",
            manifest_path=_materialize_m4_manifest(tmp_path),
        )