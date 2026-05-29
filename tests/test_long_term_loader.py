from pathlib import Path

import numpy as np

from transformer_ts_ranking.bootstrap import materialize_bootstrap_manifests
from transformer_ts_ranking.data.long_term import build_window_summary, load_long_term_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_long_term_dataset_scales_after_temporal_split(tmp_path: Path) -> None:
    repo_root = _repo_root()
    manifest_paths = materialize_bootstrap_manifests(
        repo_root=repo_root,
        config_dir=tmp_path / "configs" / "benchmark",
        audit_output_dir=tmp_path / "artifacts" / "audit",
    )

    dataset = load_long_term_dataset(
        repo_root=repo_root,
        dataset_name="ETTh1",
        manifest_path=manifest_paths["long_term_datasets"],
    )

    assert dataset.date_column == "date"
    assert dataset.feature_columns[-1] == "OT"
    assert dataset.target_columns == ["OT"]
    assert dataset.original_values.shape == dataset.scaled_values.shape
    assert dataset.time_features.shape[0] == dataset.original_values.shape[0]
    assert dataset.split_lengths["train"] == 8640
    assert dataset.split_lengths["val"] == 2880
    assert dataset.split_lengths["test"] == 2880

    restored = dataset.scaler.inverse_transform(dataset.scaled_values[:32])
    np.testing.assert_allclose(restored, dataset.original_values[:32], rtol=1e-5, atol=1e-5)


def test_build_window_summary_respects_split_boundaries(tmp_path: Path) -> None:
    repo_root = _repo_root()
    manifest_paths = materialize_bootstrap_manifests(
        repo_root=repo_root,
        config_dir=tmp_path / "configs" / "benchmark",
        audit_output_dir=tmp_path / "artifacts" / "audit",
    )

    dataset = load_long_term_dataset(
        repo_root=repo_root,
        dataset_name="ETTh1",
        manifest_path=manifest_paths["long_term_datasets"],
    )
    summary = build_window_summary(dataset=dataset, pred_len=96)

    assert summary["train"]["window_count"] == 8640 - (dataset.seq_len + 96) + 1
    assert summary["val"]["window_count"] == 2880 - (dataset.seq_len + 96) + 1
    assert summary["test"]["window_count"] == 2880 - (dataset.seq_len + 96) + 1
