from pathlib import Path

from transformer_ts_ranking.bootstrap import materialize_bootstrap_manifests
from transformer_ts_ranking.configuration import load_yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_materialize_bootstrap_manifests_writes_versioned_files(tmp_path: Path) -> None:
    repo_root = _repo_root()
    manifest_paths = materialize_bootstrap_manifests(
        repo_root=repo_root,
        config_dir=tmp_path / "configs" / "benchmark",
        audit_output_dir=tmp_path / "artifacts" / "audit",
    )

    assert set(manifest_paths) == {
        "capability_matrix",
        "long_term_datasets",
        "m4_datasets",
        "training_presets",
        "audit_inventory",
        "audit_contract",
        "audit_capability_matrix",
    }
    for path in manifest_paths.values():
        assert path.exists()

    capability_payload = load_yaml(manifest_paths["capability_matrix"])
    m4_payload = load_yaml(manifest_paths["m4_datasets"])
    assert capability_payload["manifest_kind"] == "model_capability_matrix"
    assert capability_payload["curation_status"] == "bootstrap"
    assert m4_payload["manifest_kind"] == "m4_datasets"
    assert "Yearly" in m4_payload["frequencies"]
    assert m4_payload["defaults"]["naive2_file"] == "submission-Naive2.csv"
    assert capability_payload["eligibility_policy"]["source_precedence"][0] == "manual_override"
    assert len(capability_payload["models"]) == 29
    assert capability_payload["summary"]["eligible_long_term"] == 28
    assert capability_payload["summary"]["eligible_m4"] == 28

    chronos2 = next(model for model in capability_payload["models"] if model["model_name"] == "chronos2")
    assert chronos2["eligible_long_term"] is True
    assert chronos2["eligible_m4"] is True
    assert chronos2["requires_exogenous"] is False
    assert chronos2["adapter_name"] == "encoder_only"
    assert chronos2["eligibility_source"] == "manual_override"

    airformer = next(model for model in capability_payload["models"] if model["model_name"] == "airformer")
    assert airformer["eligible_long_term"] is True
    assert airformer["eligible_m4"] is True
    assert airformer["requires_spatial_structure"] is False
    assert airformer["adapter_name"] == "encoder_only"

    tpatchgnn = next(model for model in capability_payload["models"] if model["model_name"] == "tpatchgnn")
    assert tpatchgnn["eligible_long_term"] is False
    assert tpatchgnn["eligible_m4"] is False
    assert tpatchgnn["eligibility_source"] == "manual_override"
