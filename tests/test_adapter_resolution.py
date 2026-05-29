from pathlib import Path

from transformer_ts_ranking.adapters import get_adapter_contract, iter_task_model_bindings, resolve_model_adapter
from transformer_ts_ranking.bootstrap import materialize_bootstrap_manifests
from transformer_ts_ranking.configuration import load_yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _capability_payload(tmp_path: Path) -> dict:
    manifest_paths = materialize_bootstrap_manifests(
        repo_root=_repo_root(),
        config_dir=tmp_path / "configs" / "benchmark",
        audit_output_dir=tmp_path / "artifacts" / "audit",
    )
    return load_yaml(manifest_paths["capability_matrix"])


def test_adapter_registry_exposes_expected_batch_contracts() -> None:
    contract = get_adapter_contract("seq2seq")
    task_contract = contract.for_task("long_term")

    assert contract.family_name == "seq2seq"
    assert task_contract.predict_batch_keys == ("x", "x_mark", "y_mark")
    assert "y_full" in task_contract.optional_batch_keys


def test_resolve_model_adapter_uses_capability_matrix(tmp_path: Path) -> None:
    payload = _capability_payload(tmp_path)
    binding = resolve_model_adapter("tft", capability_payload=payload)

    assert binding.model_name == "tft"
    assert binding.adapter_name == "seq2seq"
    assert binding.contract.family_name == "seq2seq"
    assert binding.eligible_long_term is True
    assert binding.eligibility_source == "manual_override"


def test_iter_task_model_bindings_excludes_tpatchgnn_from_regular_tasks(tmp_path: Path) -> None:
    payload = _capability_payload(tmp_path)

    long_term_models = {
        binding.model_name
        for binding in iter_task_model_bindings("long_term", capability_payload=payload)
    }
    m4_models = {
        binding.model_name
        for binding in iter_task_model_bindings("m4", capability_payload=payload)
    }

    assert "airformer" in long_term_models
    assert "tpatchgnn" not in long_term_models
    assert "tpatchgnn" not in m4_models