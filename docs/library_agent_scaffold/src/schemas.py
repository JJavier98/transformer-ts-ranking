"""JSON Schema generation — Component 2 (P2) of the integration design.

Turns the library's typed objects (``ForecastInput``, ``TrainingConfig``, and
each model's dataclass ``config``) into Draft-2020-12 JSON Schema so a
function-calling agent can produce valid arguments and callers can validate what
they receive. No heavy dependency — walks ``dataclasses.fields`` + type hints.

Target location in the library: ``s_transformers_lib/schemas.py``.

Tensor-typed fields (``x``, ``x_mark``, …) cannot be expressed as plain JSON, so
they are emitted as ``{shape: [...], dtype: ...}`` descriptor objects — the
contract an agent fills, with the host binding actual arrays.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any

_PRIMITIVES: dict[type, dict[str, Any]] = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
}

# Tensor-like fields are described structurally rather than inlined.
_TENSOR_SCHEMA = {
    "type": "object",
    "properties": {
        "shape": {"type": "array", "items": {"type": "integer"}},
        "dtype": {"type": "string", "default": "float32"},
    },
    "required": ["shape"],
    "description": "A tensor descriptor; the host binds the actual array.",
}
_TENSOR_FIELD_NAMES = {"x", "x_mark", "y_full", "y_mark", "x_time", "x_mask", "pred_time"}


def _type_schema(tp: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON Schema fragment."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:  # Optional[T] / T | None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _type_schema(args[0])
        return {"anyOf": [_type_schema(a) for a in args]}
    if origin in (list, tuple):
        (item,) = typing.get_args(tp) or (Any,)
        return {"type": "array", "items": _type_schema(item)}
    if tp in _PRIMITIVES:
        return dict(_PRIMITIVES[tp])
    return {}  # unknown → unconstrained


def dataclass_schema(cls: type, *, tensor_fields: set[str] | None = None) -> dict[str, Any]:
    """Return a JSON Schema object for a dataclass ``cls``.

    Parameters
    ----------
    cls : type
        A dataclass (e.g. ``ForecastInput``, ``TrainingConfig``, a model config).
    tensor_fields : set of str, optional
        Field names to emit as tensor descriptors instead of primitives.

    Returns
    -------
    dict
        A Draft-2020-12 JSON Schema ``object``.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass.")
    tensor_fields = tensor_fields or _TENSOR_FIELD_NAMES
    hints = typing.get_type_hints(cls)
    props: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(f_cls := cls):
        if f.name in tensor_fields:
            props[f.name] = dict(_TENSOR_SCHEMA)
        else:
            props[f.name] = _type_schema(hints.get(f.name, Any))
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if not has_default:
            required.append(f.name)
        elif f.default is not dataclasses.MISSING:
            props[f.name]["default"] = f.default
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f_cls.__name__,
        "type": "object",
        "properties": props,
    }
    if required:
        schema["required"] = required
    return schema


def forecast_input_schema() -> dict[str, Any]:
    """JSON Schema for ``ForecastInput`` (the ``forecast`` tool's input)."""
    from s_transformers_lib.interfaces.forecasting import ForecastInput  # noqa: PLC0415

    return dataclass_schema(ForecastInput)


def training_config_schema() -> dict[str, Any]:
    """JSON Schema for ``TrainingConfig``."""
    from s_transformers_lib.interfaces.forecasting import TrainingConfig  # noqa: PLC0415

    return dataclass_schema(TrainingConfig)


def model_config_schema(name: str) -> dict[str, Any]:
    """JSON Schema for a model's config dataclass, resolved via the registry."""
    from s_transformers_lib.models import get_config_class  # noqa: PLC0415

    return dataclass_schema(get_config_class(name))
