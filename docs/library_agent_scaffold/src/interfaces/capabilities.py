"""Intrinsic model capabilities — Component 1 (P1) of the integration design.

A model *declares* what it needs (time marks, exogenous inputs, irregular
sampling, …) so consumers never have to *guess* by introspection. This is the
single source of truth the API reference, the model cards, the model-aware
dataloaders, and the agent tools all read.

Target location in the library: ``s_transformers_lib/interfaces/capabilities.py``.

In this portable scaffold the per-model values are loaded from the bundled
``capabilities.yaml`` (extracted from the benchmark's validated capability
matrix). When adopted in the library, the recommended end state is that each
model's ``config.py`` declares its own ``ModelCapabilities`` and the registry
aggregates them; ``load_capabilities`` then becomes a thin fallback. Either way
the public API below is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Callable

import yaml

# The capabilities YAML sits next to this package in the scaffold; in the library
# it would live beside the models or be assembled from per-model declarations.
_CAPS_FILE = Path(__file__).resolve().parent.parent / "capabilities.yaml"


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model intrinsically requires and supports.

    Attributes
    ----------
    supports_regular_mts : bool
        Handles regularly-sampled multivariate time series.
    supports_univariate : bool
        Handles single-channel series.
    requires_time_marks : bool
        Needs calendar/time-feature tensors (``x_mark`` / ``y_mark``).
    requires_exogenous : bool
        Needs exogenous covariates.
    requires_irregular_times : bool
        Needs irregular-sampling inputs (``x_time`` / ``x_mask`` / ``pred_time``).
    requires_spatial_structure : bool
        Needs an explicit spatial/graph structure.
    is_pretrained_zeroshot : bool
        Ships pretrained weights and can forecast with little or no training.
    family : str
        Batch/architecture family: ``"encoder_only"``, ``"seq2seq"``,
        ``"exogenous_aware"``, ``"irregular"``, ``"pretrained_zeroshot"``.
    """

    supports_regular_mts: bool
    supports_univariate: bool
    requires_time_marks: bool
    requires_exogenous: bool
    requires_irregular_times: bool
    requires_spatial_structure: bool
    is_pretrained_zeroshot: bool
    family: str


@lru_cache(maxsize=1)
def load_capabilities() -> dict[str, ModelCapabilities]:
    """Return ``{model_name: ModelCapabilities}`` for every known model."""
    raw = yaml.safe_load(_CAPS_FILE.read_text())
    valid = {f.name for f in fields(ModelCapabilities)}
    return {
        name: ModelCapabilities(**{k: v for k, v in spec.items() if k in valid})
        for name, spec in raw.items()
    }


def capabilities(name: str) -> ModelCapabilities:
    """Return the :class:`ModelCapabilities` for ``name``.

    Parameters
    ----------
    name : str
        Canonical model key (as in ``list_models()``).

    Returns
    -------
    ModelCapabilities

    Raises
    ------
    KeyError
        If ``name`` has no declared capabilities.
    """
    caps = load_capabilities()
    if name not in caps:
        raise KeyError(f"No declared capabilities for model {name!r}.")
    return caps[name]


def filter_models(predicate: Callable[[ModelCapabilities], bool]) -> list[str]:
    """Return model names whose capabilities satisfy ``predicate``.

    Examples
    --------
    >>> filter_models(lambda c: c.supports_regular_mts and not c.requires_irregular_times)
    """
    return sorted(n for n, c in load_capabilities().items() if predicate(c))
