"""Capability-matrix driven adapter resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..configuration import load_yaml
from .base import ModelAdapterBinding, TaskName
from .registry import get_adapter_contract


def _load_capability_payload(capability_path: Path) -> dict[str, Any]:
    """Load the versioned capability matrix from disk.

    Args:
        capability_path: Path to ``model_capability_matrix.yaml``.

    Returns:
        Parsed capability payload.
    """
    return load_yaml(capability_path)


def _entry_by_model_name(capability_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index capability entries by model name for deterministic lookups.

    Args:
        capability_payload: Full capability matrix payload.

    Returns:
        A mapping from model name to its capability entry.
    """
    return {
        entry["model_name"]: entry
        for entry in capability_payload.get("models", [])
    }


def resolve_model_adapter(
    model_name: str,
    capability_payload: dict[str, Any] | None = None,
    capability_path: Path | None = None,
) -> ModelAdapterBinding:
    """Resolve one model to its adapter contract using the capability matrix.

    Args:
        model_name: Model identifier to resolve.
        capability_payload: Optional capability payload already loaded in memory.
        capability_path: Optional YAML path used when the payload is not provided.

    Returns:
        The resolved model-to-adapter binding.
    """
    if capability_payload is None:
        if capability_path is None:
            raise ValueError("resolve_model_adapter() requires capability_payload or capability_path.")
        capability_payload = _load_capability_payload(capability_path)

    entries = _entry_by_model_name(capability_payload)
    if model_name not in entries:
        available = ", ".join(sorted(entries))
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

    entry = entries[model_name]
    adapter_name = entry.get("adapter_name")
    if not adapter_name:
        raise ValueError(f"Model '{model_name}' is missing adapter_name in the capability matrix.")

    # The returned object keeps both normalized contract data and the raw matrix
    # entry so future runners can use either view without reloading YAML.
    return ModelAdapterBinding(
        model_name=model_name,
        adapter_name=adapter_name,
        contract=get_adapter_contract(adapter_name),
        eligible_long_term=bool(entry.get("eligible_long_term")),
        eligible_m4=bool(entry.get("eligible_m4")),
        review_status=str(entry.get("review_status")),
        eligibility_source=str(entry.get("eligibility_source")),
        eligibility_reason=str(entry.get("eligibility_reason")),
        supports_regular_mts=entry.get("supports_regular_mts"),
        supports_univariate=entry.get("supports_univariate"),
        notes=tuple(entry.get("notes", [])),
        raw_capability=entry,
    )


def iter_task_model_bindings(
    task_name: TaskName,
    capability_payload: dict[str, Any] | None = None,
    capability_path: Path | None = None,
) -> list[ModelAdapterBinding]:
    """Return all capability entries resolved to adapter contracts for one task.

    Args:
        task_name: Benchmark task to filter by.
        capability_payload: Optional capability payload already loaded in memory.
        capability_path: Optional YAML path used when the payload is not provided.

    Returns:
        All resolved bindings eligible for the requested task.
    """
    if capability_payload is None:
        if capability_path is None:
            raise ValueError("iter_task_model_bindings() requires capability_payload or capability_path.")
        capability_payload = _load_capability_payload(capability_path)

    bindings = [
        resolve_model_adapter(
            model_name=entry["model_name"],
            capability_payload=capability_payload,
        )
        for entry in capability_payload.get("models", [])
    ]
    return [binding for binding in bindings if binding.is_eligible_for(task_name)]