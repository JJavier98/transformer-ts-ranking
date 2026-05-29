"""Smoke runners for validating benchmark manifests and data plumbing.

These runners validate the pre-training stages of the benchmark: manifests,
dataset loading, adapter resolution and metric wiring. They intentionally stop
short of exhaustive training so the repository can be checked quickly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from ..adapters import iter_task_model_bindings, resolve_model_adapter
from ..adapters.base import TaskName
from ..bootstrap import materialize_bootstrap_manifests
from ..configuration import load_yaml, write_json
from ..data.long_term import build_window_summary, load_long_term_dataset
from ..data.m4 import load_m4_dataset
from .m4_metrics import evaluate_m4_dataset

__all__ = ["run_long_term_smoke", "run_m4_smoke"]


def _ensure_manifests(repo_root: Path, config_dir: Path) -> dict[str, Path]:
    """Materialize manifests on demand so the smoke runner is self-contained.

    Args:
        repo_root: Root directory of the benchmark repository.
        config_dir: Directory where versioned manifests should exist.

    Returns:
        Paths to the manifest files required by the smoke runner.
    """
    capability_path = config_dir / "model_capability_matrix.yaml"
    datasets_path = config_dir / "long_term_datasets.yaml"
    m4_path = config_dir / "m4_datasets.yaml"
    presets_path = config_dir / "training_presets.yaml"

    if capability_path.exists() and datasets_path.exists() and m4_path.exists() and presets_path.exists():
        return {
            "capability_matrix": capability_path,
            "long_term_datasets": datasets_path,
            "m4_datasets": m4_path,
            "training_presets": presets_path,
        }

    return materialize_bootstrap_manifests(
        repo_root=repo_root,
        config_dir=config_dir,
        audit_output_dir=repo_root / "artifacts" / "audit",
    )


def _check_runtime_dependencies() -> dict[str, Any]:
    """Attempt a lightweight Torch import and keep the smoke run non-fatal.

    Returns:
        A small status payload describing whether the runtime is available.
    """
    try:
        _ = import_module("torch")
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on local runtime
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "reason": None,
    }


def _select_smoke_models(
    capability_payload: dict[str, Any],
    preset_payload: dict[str, Any],
    task_name: TaskName,
) -> list[str]:
    """Pick representative eligible models for one benchmark task.

    Args:
        capability_payload: Versioned capability matrix.
        preset_payload: Training preset manifest.
        task_name: Benchmark task name.

    Returns:
        A short, representative list of eligible models for smoke checks.
    """
    preset = preset_payload["presets"]["smoke"]
    preferred_models = list(preset.get("representative_models", []))
    eligible_models = {
        binding.model_name: binding
        for binding in iter_task_model_bindings(
            task_name=task_name,
            capability_payload=capability_payload,
        )
    }

    selected_models: list[str] = []
    for model_name in preferred_models:
        if model_name in eligible_models:
            selected_models.append(model_name)
    if len(selected_models) >= int(preset["max_models"]):
        return selected_models[: int(preset["max_models"])]

    seen_adapters = {eligible_models[model_name].adapter_name for model_name in selected_models}
    for model_name, binding in eligible_models.items():
        if model_name in selected_models:
            continue
        adapter_name = binding.adapter_name
        if adapter_name not in seen_adapters:
            # Adapter diversity matters more than model count in smoke mode because
            # it exercises different batch contracts with minimal runtime cost.
            selected_models.append(model_name)
            seen_adapters.add(adapter_name)
        if len(selected_models) >= int(preset["max_models"]):
            return selected_models

    for model_name in eligible_models:
        if model_name not in selected_models:
            selected_models.append(model_name)
        if len(selected_models) >= int(preset["max_models"]):
            break
    return selected_models


def run_long_term_smoke(
    repo_root: Path,
    dataset_name: str,
    preset_name: str,
    config_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Build a smoke-plan artifact for one long-term dataset.

    This runner validates the data path, the manifests and the model selection
    logic. It deliberately stops short of training because the local runtime may
    not yet satisfy Torch/CUDA requirements.

    Args:
        repo_root: Root directory of the benchmark repository.
        dataset_name: Long-term dataset key to inspect.
        preset_name: Training preset used to decide smoke scope.
        config_dir: Directory containing versioned manifests.
        output_dir: Directory where the smoke artifact will be written.

    Returns:
        The smoke payload and the path of the generated JSON artifact.
    """
    repo_root = repo_root.resolve()
    config_dir = config_dir.resolve()
    output_dir = output_dir.resolve()

    manifest_paths = _ensure_manifests(repo_root=repo_root, config_dir=config_dir)
    capability_payload = load_yaml(manifest_paths["capability_matrix"])
    preset_payload = load_yaml(manifest_paths["training_presets"])
    dataset_payload = load_yaml(manifest_paths["long_term_datasets"])

    if preset_name not in preset_payload["presets"]:
        available = ", ".join(sorted(preset_payload["presets"]))
        raise KeyError(f"Unknown preset '{preset_name}'. Available: {available}")

    dataset = load_long_term_dataset(
        repo_root=repo_root,
        dataset_name=dataset_name,
        manifest_path=manifest_paths["long_term_datasets"],
    )
    runtime_status = _check_runtime_dependencies()
    selected_models = _select_smoke_models(
        capability_payload=capability_payload,
        preset_payload=preset_payload,
        task_name="long_term",
    )
    selected_adapters = [
        resolve_model_adapter(
            model_name=model_name,
            capability_payload=capability_payload,
        )
        for model_name in selected_models
    ]
    preset = preset_payload["presets"][preset_name]
    selected_horizons = dataset.horizons[: int(preset["max_horizons_per_dataset"])]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset_name": preset_name,
        "runtime": runtime_status,
        "selected_models": selected_models,
        "selected_adapters": [
            {
                "model_name": binding.model_name,
                "adapter_name": binding.adapter_name,
                "description": binding.contract.description,
                "predict_batch_keys": list(binding.contract.for_task("long_term").predict_batch_keys),
                "optional_batch_keys": list(binding.contract.for_task("long_term").optional_batch_keys),
                "review_status": binding.review_status,
            }
            for binding in selected_adapters
        ],
        "dataset": {
            "dataset_name": dataset.dataset_name,
            "source_path": str(dataset.source_path),
            "frequency": dataset.frequency,
            "rows": len(dataset.frame),
            "feature_count": len(dataset.feature_columns),
            "feature_columns": dataset.feature_columns,
            "target_columns": dataset.target_columns,
            "split_lengths": dataset.split_lengths,
            "time_feature_dim": int(dataset.time_features.shape[1]),
            "available_horizons": dataset.horizons,
            "selected_horizons": selected_horizons,
        },
        "windows": {
            str(horizon): build_window_summary(dataset=dataset, pred_len=horizon)
            for horizon in selected_horizons
        },
        "manifests": {
            "config_dir": str(config_dir),
            "capability_matrix": str(manifest_paths["capability_matrix"]),
            "long_term_datasets": str(manifest_paths["long_term_datasets"]),
            "training_presets": str(manifest_paths["training_presets"]),
            "dataset_manifest_kind": dataset_payload.get("manifest_kind"),
        },
    }

    artifact_path = write_json(
        payload,
        output_dir / f"long_term_smoke_{dataset_name}_{preset_name}.json",
    )
    return payload, artifact_path


def run_m4_smoke(
    repo_root: Path,
    frequency_label: str,
    preset_name: str,
    config_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Build a smoke-plan artifact for one M4 frequency slice.

    Args:
        repo_root: Root directory of the benchmark repository.
        frequency_label: M4 frequency label to inspect.
        preset_name: Training preset used to decide smoke scope.
        config_dir: Directory containing versioned manifests.
        output_dir: Directory where the smoke artifact will be written.

    Returns:
        The smoke payload and the path of the generated JSON artifact.
    """
    repo_root = repo_root.resolve()
    config_dir = config_dir.resolve()
    output_dir = output_dir.resolve()

    manifest_paths = _ensure_manifests(repo_root=repo_root, config_dir=config_dir)
    capability_payload = load_yaml(manifest_paths["capability_matrix"])
    preset_payload = load_yaml(manifest_paths["training_presets"])
    dataset_payload = load_yaml(manifest_paths["m4_datasets"])

    if preset_name not in preset_payload["presets"]:
        available = ", ".join(sorted(preset_payload["presets"]))
        raise KeyError(f"Unknown preset '{preset_name}'. Available: {available}")

    dataset = load_m4_dataset(
        repo_root=repo_root,
        frequency_label=frequency_label,
        manifest_path=manifest_paths["m4_datasets"],
    )
    runtime_status = _check_runtime_dependencies()
    selected_models = _select_smoke_models(
        capability_payload=capability_payload,
        preset_payload=preset_payload,
        task_name="m4",
    )
    selected_adapters = [
        resolve_model_adapter(
            model_name=model_name,
            capability_payload=capability_payload,
        )
        for model_name in selected_models
    ]

    # Evaluating Naive2 through the same metric path proves the M4 loader and
    # metric implementation are aligned before real model forecasts are added.
    naive2_predictions = {series_id: series.naive2_forecast for series_id, series in dataset.series.items()}
    naive2_reference = evaluate_m4_dataset(dataset=dataset, predictions=naive2_predictions)

    category_counts: dict[str, int] = {}
    for series in dataset.series.values():
        category_counts[series.category] = category_counts.get(series.category, 0) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset_name": preset_name,
        "runtime": runtime_status,
        "selected_models": selected_models,
        "selected_adapters": [
            {
                "model_name": binding.model_name,
                "adapter_name": binding.adapter_name,
                "description": binding.contract.description,
                "predict_batch_keys": list(binding.contract.for_task("m4").predict_batch_keys),
                "optional_batch_keys": list(binding.contract.for_task("m4").optional_batch_keys),
                "review_status": binding.review_status,
            }
            for binding in selected_adapters
        ],
        "dataset": {
            "frequency_label": dataset.frequency_label,
            "frequency_code": dataset.frequency_code,
            "horizon": dataset.horizon,
            "series_count": dataset.series_count,
            "sample_series_ids": dataset.series_ids[:5],
            "category_counts": category_counts,
        },
        "naive2_reference": {
            "mean_smape": naive2_reference.mean_smape,
            "mean_mase": naive2_reference.mean_mase,
            "mean_owa": naive2_reference.mean_owa,
        },
        "manifests": {
            "config_dir": str(config_dir),
            "capability_matrix": str(manifest_paths["capability_matrix"]),
            "m4_datasets": str(manifest_paths["m4_datasets"]),
            "training_presets": str(manifest_paths["training_presets"]),
            "dataset_manifest_kind": dataset_payload.get("manifest_kind"),
        },
    }

    artifact_path = write_json(
        payload,
        output_dir / f"m4_smoke_{frequency_label}_{preset_name}.json",
    )
    return payload, artifact_path
