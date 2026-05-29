"""Runtime compatibility probes for canonical benchmark execution.

All checks in this module use only canonical forecast fields and do not apply
legacy batch wrappers or per-model fallbacks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Iterator

import torch
import torch.nn as nn

from ..configuration import write_json

__all__ = [
    "probe_model_compatibility",
    "validate_canonical_forward_pass",
]


@dataclass(frozen=True)
class TaskSpec:
    """Synthetic task definition used by the runtime compatibility probe."""

    name: str
    seq_len: int
    label_len: int
    pred_len: int
    enc_in: int
    c_out: int
    n_time_features: int


@dataclass
class TaskProbeResult:
    """Per-task execution result for one model."""

    task_name: str
    compatible: bool
    fit_ok: bool
    fit_error: str | None
    output_container: str | None
    successful_profile: str | None
    output_shape: list[int] | None
    eval_step_ok: bool
    error: str | None
    attempted_profiles: list[str]
    config_snapshot: dict[str, Any]


@dataclass
class ModelProbeResult:
    """Compatibility evidence collected for one model across both tasks."""

    model_name: str
    long_term: TaskProbeResult
    m4: TaskProbeResult


@dataclass
class ForwardTaskResult:
    """Per-task forward-pass validation result for one model."""

    task_name: str
    forward_ok: bool
    output_container: str | None
    output_shape: list[int] | None
    error: str | None
    config_snapshot: dict[str, Any]


@dataclass
class ForwardModelResult:
    """Forward validation evidence collected for one model across both tasks."""

    model_name: str
    long_term: ForwardTaskResult
    m4: ForwardTaskResult


TASK_SPECS = {
    "long_term": TaskSpec(
        name="long_term",
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=7,
        c_out=7,
        n_time_features=4,
    ),
    "m4": TaskSpec(
        name="m4",
        seq_len=96,
        label_len=48,
        pred_len=24,
        enc_in=1,
        c_out=1,
        n_time_features=4,
    ),
}


SPECIAL_TASK_CONFIG_OVERRIDES = {
    "earthformer": {
        "long_term": {"data_mode": "1d", "input_height": 1, "input_width": 1},
        "m4": {"data_mode": "1d", "input_height": 1, "input_width": 1},
    },
}


@contextmanager
def _prepended_sys_path(path: Path) -> Iterator[None]:
    """Temporarily prepend a path so the probe can import the submodule package.

    Args:
        path: Path to prepend temporarily to ``sys.path``.

    Yields:
        ``None`` while the import path is active.
    """
    path_str = str(path)
    original = list(sys.path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        sys.path[:] = original


def _set_attr_if_present(config: Any, name: str, value: Any) -> None:
    """Write a config attribute only when the target config exposes it.

    Args:
        config: Model config instance to update.
        name: Attribute name to set.
        value: Value to assign when the attribute exists.
    """
    if hasattr(config, name):
        setattr(config, name, value)


def _configure_for_task(model_name: str, config: Any, task: TaskSpec) -> Any:
    """Apply generic and model-specific overrides for one benchmark task.

    Args:
        model_name: Model identifier under review.
        config: Fresh config instance to adapt.
        task: Synthetic task specification used by the probe.

    Returns:
        The mutated config instance ready for model construction.
    """
    _set_attr_if_present(config, "seq_len", task.seq_len)
    _set_attr_if_present(config, "label_len", task.label_len)
    _set_attr_if_present(config, "pred_len", task.pred_len)
    _set_attr_if_present(config, "enc_in", task.enc_in)
    _set_attr_if_present(config, "dec_in", task.enc_in)
    _set_attr_if_present(config, "c_out", task.c_out)
    _set_attr_if_present(config, "n_time_features", task.n_time_features)
    _set_attr_if_present(config, "d_mark", task.n_time_features)
    _set_attr_if_present(config, "d_x", task.n_time_features)
    _set_attr_if_present(config, "n_future_covariates", 0)
    _set_attr_if_present(config, "use_gradient_checkpointing", False)

    for key, value in SPECIAL_TASK_CONFIG_OVERRIDES.get(model_name, {}).get(task.name, {}).items():
        _set_attr_if_present(config, key, value)

    # Some configs enforce invariants in validate or __post_init__, so both hooks
    # are replayed after the synthetic task overrides are applied.
    if hasattr(config, "validate") and callable(config.validate):
        config.validate()
    if hasattr(config, "__post_init__") and callable(config.__post_init__):
        config.__post_init__()
    return config


def _config_snapshot(config: Any) -> dict[str, Any]:
    """Serialize the relevant config values for the compatibility report.

    Args:
        config: Config instance used to build the model.

    Returns:
        A JSON-serializable snapshot of the config.
    """
    if hasattr(config, "to_dict") and callable(config.to_dict):
        payload = config.to_dict()
        if isinstance(payload, dict):
            return payload
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"repr": repr(config)}


def _canonical_probe_batch(
    task: TaskSpec,
    *,
    batch_size: int,
    include_target: bool,
) -> dict[str, torch.Tensor]:
    """Create one synthetic canonical batch for training/evaluation/inference."""
    x = torch.randn(batch_size, task.seq_len, task.enc_in)
    batch: dict[str, torch.Tensor] = {
        "x": x,
        "x_mark": torch.randn(batch_size, task.seq_len, task.n_time_features),
        "y_mark": torch.randn(batch_size, task.label_len + task.pred_len, task.n_time_features),
    }
    if include_target:
        batch["y"] = torch.randn(batch_size, task.pred_len, task.c_out)
    return batch


def _fit_probe_batch(task: TaskSpec) -> dict[str, torch.Tensor]:
    """Create a compact train batch for one ``fit()`` probe call.

    Args:
        task: Synthetic task specification.

    Returns:
        A canonical batch dictionary accepted by the new sklearn-like API.
    """
    # Use batch size 1 to keep API checks lightweight even for large models.
    return _canonical_probe_batch(task, batch_size=1, include_target=True)


def _extract_tensor_output(output: Any) -> torch.Tensor:
    """Normalize model outputs so the probe can validate shapes uniformly.

    Args:
        output: Object returned by ``model.predict``.

    Returns:
        The main tensor forecast.
    """
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor output, got {type(output).__name__}")
    return output


def _run_fit_probe(model: Any, task: TaskSpec) -> tuple[bool, str | None]:
    """Run a minimal ``fit()`` call to validate the public training API.

    Args:
        model: Instantiated model.
        task: Synthetic task definition.

    Returns:
        ``(fit_ok, error_message)`` where ``error_message`` is populated on failure.
    """
    train_batch = _fit_probe_batch(task)
    try:
        model.fit(
            train_data=train_batch,
            training={"epochs": 1, "batch_size": None, "device": "cpu", "verbose": False},
        )
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return False, f"fit() failed: {type(exc).__name__}: {exc}"
    return True, None


def _probe_task(model: Any, config: Any, task: TaskSpec) -> TaskProbeResult:
    """Run ``predict()`` and ``eval_step()`` against one synthetic task.

    Args:
        model: Instantiated model.
        config: Config used to build the model.
        task: Synthetic task specification.

    Returns:
        Compatibility evidence for the requested task.
    """
    canonical_batch = _canonical_probe_batch(task, batch_size=1, include_target=True)
    attempted_profiles = ["canonical"]
    loss_fn = nn.MSELoss()
    expected_shape = [1, task.pred_len, task.c_out]
    error_message = "Canonical probe failed."
    fit_ok, fit_error = _run_fit_probe(model=model, task=task)
    if not fit_ok and fit_error is not None:
        error_message = fit_error
    try:
        predict_output = model.predict(canonical_batch, device="cpu")
        output_container = type(predict_output).__name__
        if hasattr(predict_output, "prediction"):
            output = _extract_tensor_output(predict_output.prediction)
        else:
            output = _extract_tensor_output(predict_output)
        if list(output.shape) != expected_shape:
            raise ValueError(
                f"Unexpected output shape {list(output.shape)}; expected {expected_shape}"
            )
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        return TaskProbeResult(
            task_name=task.name,
            compatible=False,
            fit_ok=fit_ok,
            fit_error=fit_error,
            output_container=None,
            successful_profile=None,
            output_shape=None,
            eval_step_ok=False,
            error=error_message,
            attempted_profiles=attempted_profiles,
            config_snapshot=_config_snapshot(config),
        )

    eval_step_ok = True
    eval_error: str | None = None
    try:
        model.eval_step(canonical_batch, loss_fn, device="cpu")
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        # ``eval_step`` is not part of the strict canonical inference contract.
        # Keep compatibility focused on fit()/predict() while still reporting
        # eval_step issues for follow-up hardening.
        eval_step_ok = False
        eval_error = f"eval_step failed: {type(exc).__name__}: {exc}"

    return TaskProbeResult(
        task_name=task.name,
        compatible=True,
        fit_ok=fit_ok,
        fit_error=fit_error,
        output_container=output_container,
        successful_profile="canonical",
        output_shape=list(output.shape),
        eval_step_ok=eval_step_ok,
        error=eval_error,
        attempted_profiles=attempted_profiles,
        config_snapshot=_config_snapshot(config),
    )


def _forward_validate_task(model: Any, config: Any, task: TaskSpec) -> ForwardTaskResult:
    """Validate one canonical forward pass through predict()."""
    batch = _canonical_probe_batch(task, batch_size=1, include_target=False)
    expected_shape = [1, task.pred_len, task.c_out]
    try:
        predict_output = model.predict(batch, device="cpu")
        output_container = type(predict_output).__name__
        if hasattr(predict_output, "prediction"):
            output = _extract_tensor_output(predict_output.prediction)
        else:
            output = _extract_tensor_output(predict_output)
        if list(output.shape) != expected_shape:
            raise ValueError(
                f"Unexpected output shape {list(output.shape)}; expected {expected_shape}"
            )
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return ForwardTaskResult(
            task_name=task.name,
            forward_ok=False,
            output_container=None,
            output_shape=None,
            error=f"{type(exc).__name__}: {exc}",
            config_snapshot=_config_snapshot(config),
        )

    return ForwardTaskResult(
        task_name=task.name,
        forward_ok=True,
        output_container=output_container,
        output_shape=list(output.shape),
        error=None,
        config_snapshot=_config_snapshot(config),
    )


def probe_model_compatibility(
    repo_root: Path,
    output_path: Path,
    models: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Probe model compatibility for the long-term and M4 benchmark tasks.

    Args:
        repo_root: Root directory of the benchmark repository.
        output_path: JSON artifact path for the compatibility report.
        models: Optional subset of model names. When omitted, all registered models are probed.

    Returns:
        The compatibility payload and the path where it was written.
    """
    repo_root = repo_root.resolve()
    submodule_root = repo_root / "s-transformers-lib"
    output_path = output_path.resolve()

    with _prepended_sys_path(submodule_root):
        registry_module = import_module("src.models.registry")
        create_model = registry_module.create_model
        get_config_class = registry_module.get_config_class
        list_models = registry_module.list_models

        model_names = models or list_models()
        results: list[ModelProbeResult] = []

        for model_name in model_names:
            config_cls = get_config_class(model_name)

            # Each task gets a fresh config/model pair so task-specific mutations
            # cannot leak from long_term into m4 or vice versa.
            long_term_config = _configure_for_task(model_name, config_cls(), TASK_SPECS["long_term"])
            long_term_model = create_model(model_name, long_term_config)
            long_term_result = _probe_task(
                model=long_term_model,
                config=long_term_config,
                task=TASK_SPECS["long_term"],
            )

            m4_config = _configure_for_task(model_name, config_cls(), TASK_SPECS["m4"])
            m4_model = create_model(model_name, m4_config)
            m4_result = _probe_task(
                model=m4_model,
                config=m4_config,
                task=TASK_SPECS["m4"],
            )

            results.append(
                ModelProbeResult(
                    model_name=model_name,
                    long_term=long_term_result,
                    m4=m4_result,
                )
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "summary": {
            "total_models": len(results),
            "long_term_compatible": sum(1 for result in results if result.long_term.compatible),
            "m4_compatible": sum(1 for result in results if result.m4.compatible),
            "long_term_fit_ok": sum(1 for result in results if result.long_term.fit_ok),
            "m4_fit_ok": sum(1 for result in results if result.m4.fit_ok),
            "both_compatible": sum(
                1
                for result in results
                if result.long_term.compatible and result.m4.compatible
            ),
            "both_fit_ok": sum(
                1
                for result in results
                if result.long_term.fit_ok and result.m4.fit_ok
            ),
        },
        "results": [asdict(result) for result in results],
    }
    artifact_path = write_json(payload, output_path)
    return payload, artifact_path


def validate_canonical_forward_pass(
    repo_root: Path,
    output_path: Path,
    models: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Validate predict() forward passes using canonical inputs only.

    Args:
        repo_root: Root directory of the benchmark repository.
        output_path: JSON artifact path for the forward validation report.
        models: Optional subset of model names. When omitted, all registered models are validated.

    Returns:
        The validation payload and the path where it was written.
    """
    repo_root = repo_root.resolve()
    submodule_root = repo_root / "s-transformers-lib"
    output_path = output_path.resolve()

    with _prepended_sys_path(submodule_root):
        registry_module = import_module("src.models.registry")
        create_model = registry_module.create_model
        get_config_class = registry_module.get_config_class
        list_models = registry_module.list_models

        model_names = models or list_models()
        results: list[ForwardModelResult] = []

        for model_name in model_names:
            config_cls = get_config_class(model_name)

            long_term_config = _configure_for_task(model_name, config_cls(), TASK_SPECS["long_term"])
            long_term_model = create_model(model_name, long_term_config)
            long_term_result = _forward_validate_task(
                model=long_term_model,
                config=long_term_config,
                task=TASK_SPECS["long_term"],
            )

            m4_config = _configure_for_task(model_name, config_cls(), TASK_SPECS["m4"])
            m4_model = create_model(model_name, m4_config)
            m4_result = _forward_validate_task(
                model=m4_model,
                config=m4_config,
                task=TASK_SPECS["m4"],
            )

            results.append(
                ForwardModelResult(
                    model_name=model_name,
                    long_term=long_term_result,
                    m4=m4_result,
                )
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "summary": {
            "total_models": len(results),
            "long_term_forward_ok": sum(1 for result in results if result.long_term.forward_ok),
            "m4_forward_ok": sum(1 for result in results if result.m4.forward_ok),
            "both_forward_ok": sum(
                1
                for result in results
                if result.long_term.forward_ok and result.m4.forward_ok
            ),
        },
        "results": [asdict(result) for result in results],
    }
    artifact_path = write_json(payload, output_path)
    return payload, artifact_path