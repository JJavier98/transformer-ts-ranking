"""Model selection — Component 5 (P6) of the integration design.

Answers the highest-value agent question — *which model?* — from a task
descriptor, using the declared capabilities (and, optionally, published
benchmark ranks). Pure capability logic; no model imports needed.

Target location in the library: ``s_transformers_lib/selection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interfaces.capabilities import ModelCapabilities, load_capabilities


@dataclass(frozen=True)
class TaskDescriptor:
    """A forecasting task, as an agent would describe it."""

    horizon: int
    multivariate: bool = True
    exogenous: bool = False
    irregular: bool = False
    univariate: bool = False


def _eligible(caps: ModelCapabilities, task: TaskDescriptor) -> bool:
    """Hard feasibility filter: can this model run this task at all?"""
    if task.irregular and not caps.requires_irregular_times:
        return False
    if not task.irregular and caps.requires_irregular_times:
        return False  # irregular-only models on regular data
    if task.exogenous and not (caps.requires_exogenous or caps.family == "exogenous_aware"):
        return False
    if task.univariate and not caps.supports_univariate:
        return False
    if task.multivariate and not caps.supports_regular_mts:
        return False
    return True


def recommend_models(
    task: TaskDescriptor,
    *,
    ranks: dict[str, float] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return ranked candidate models for ``task`` with a short reason each.

    Parameters
    ----------
    task : TaskDescriptor
        The task to match.
    ranks : dict, optional
        ``{model_name: mean_rank}`` from the published benchmark leaderboard
        (lower = better). When given, feasible candidates are ordered by it.
    top_k : int
        Maximum number of candidates to return.

    Returns
    -------
    list of dict
        ``[{"name": ..., "why": ...}, ...]``, best first.
    """
    feasible = [
        (name, caps)
        for name, caps in load_capabilities().items()
        if _eligible(caps, task)
    ]

    def sort_key(item: tuple[str, ModelCapabilities]) -> tuple[float, str]:
        name, _ = item
        return (ranks.get(name, float("inf")) if ranks else 0.0, name)

    feasible.sort(key=sort_key)

    out: list[dict[str, Any]] = []
    for name, caps in feasible[:top_k]:
        why = f"family={caps.family}"
        if ranks and name in ranks:
            why += f", benchmark mean-rank={ranks[name]:.2f}"
        out.append({"name": name, "why": why})
    return out
