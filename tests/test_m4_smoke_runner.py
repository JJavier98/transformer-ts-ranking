from pathlib import Path

from transformer_ts_ranking.evaluation.smoke import run_m4_smoke


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_m4_smoke_writes_plan_artifact(tmp_path: Path) -> None:
    payload, artifact_path = run_m4_smoke(
        repo_root=_repo_root(),
        frequency_label="Hourly",
        preset_name="smoke",
        config_dir=tmp_path / "configs" / "benchmark",
        output_dir=tmp_path / "artifacts" / "smoke",
    )

    assert payload["dataset"]["frequency_label"] == "Hourly"
    assert payload["dataset"]["horizon"] == 48
    assert payload["selected_models"]
    assert payload["selected_models_metadata"]
    assert payload["naive2_reference"]["mean_owa"] == 1.0
    assert artifact_path.exists()