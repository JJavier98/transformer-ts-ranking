from pathlib import Path

from transformer_ts_ranking.discovery.model_audit import (
    audit_model_library,
    write_audit_artifacts,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_audit_model_library_has_no_inventory_drift() -> None:
    repo_root = _repo_root()
    report = audit_model_library(repo_root=repo_root, submodule_root=repo_root / "s-transformers-lib")

    assert report.summary["filesystem_only"] == 0
    assert report.summary["registry_only"] == 0
    assert report.summary["inspection_failures"] == 0


def test_audit_model_library_resolves_tft_alias() -> None:
    repo_root = _repo_root()
    report = audit_model_library(repo_root=repo_root, submodule_root=repo_root / "s-transformers-lib")

    tft_entry = next(model for model in report.models if model.model_name == "tft")

    assert tft_entry.source_inspected is True
    assert tft_entry.model_class_name == "TFT"
    assert "x_mark_dec" in tft_entry.forward_params
    assert tft_entry.adapter_name == "exogenous_aware"


def test_write_audit_artifacts_creates_expected_files(tmp_path: Path) -> None:
    repo_root = _repo_root()
    report = audit_model_library(repo_root=repo_root, submodule_root=repo_root / "s-transformers-lib")

    artifact_paths = write_audit_artifacts(report=report, output_dir=tmp_path)

    assert set(artifact_paths) == {"inventory", "contract", "capability_matrix"}
    for path in artifact_paths.values():
        assert path.exists()