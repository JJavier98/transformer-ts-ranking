from pathlib import Path

from transformer_ts_ranking.evaluation.smoke import run_long_term_smoke


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_long_term_smoke_writes_plan_artifact(tmp_path: Path) -> None:
    repo_root = _repo_root()
    payload, artifact_path = run_long_term_smoke(
        repo_root=repo_root,
        dataset_name="ETTh1",
        preset_name="smoke",
        config_dir=tmp_path / "configs" / "benchmark",
        output_dir=tmp_path / "artifacts" / "smoke",
    )

    assert payload["dataset"]["dataset_name"] == "ETTh1"
    assert payload["selected_models"]
    assert payload["selected_adapters"]
    assert len(payload["selected_adapters"]) == len(payload["selected_models"])
    assert payload["dataset"]["selected_horizons"] == [96]
    assert payload["windows"]["96"]["train"]["window_count"] > 0
    assert artifact_path.exists()