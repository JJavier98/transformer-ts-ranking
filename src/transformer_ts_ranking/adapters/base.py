"""Core adapter dataclasses used by the benchmark runner.

The current slice does not instantiate models yet. It formalizes the adapter
families declared in the capability matrix so future runners can resolve model
handling from manifests instead of hard-coded name checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TaskName = Literal["long_term", "m4"]


@dataclass(frozen=True)
class AdapterTaskContract:
    """Task-specific batch contract for one adapter family.

    Attributes:
        task_name: Benchmark task where the contract applies.
        train_batch_keys: Required keys for training batches.
        eval_batch_keys: Required keys for validation or test batches.
        predict_batch_keys: Required keys for pure inference.
        optional_batch_keys: Supported but non-mandatory keys.
        notes: Extra caveats about the family-task combination.
    """

    task_name: TaskName
    train_batch_keys: tuple[str, ...]
    eval_batch_keys: tuple[str, ...]
    predict_batch_keys: tuple[str, ...]
    optional_batch_keys: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True)
class AdapterContract:
    """Benchmark-facing contract for one adapter family.

    Attributes:
        family_name: Stable family identifier stored in the capability matrix.
        description: Human-readable description of the family.
        task_contracts: Per-task batch contracts.
    """

    family_name: str
    description: str
    task_contracts: dict[TaskName, AdapterTaskContract]

    def for_task(self, task_name: TaskName) -> AdapterTaskContract:
        """Return the task-specific batch contract for this adapter family.

        Args:
            task_name: Benchmark task to resolve.

        Returns:
            The task-specific contract for the adapter family.
        """
        if task_name not in self.task_contracts:
            raise KeyError(f"Adapter family '{self.family_name}' does not define task '{task_name}'.")
        return self.task_contracts[task_name]


@dataclass(frozen=True)
class ModelAdapterBinding:
    """Capability-matrix entry enriched with its resolved adapter contract.

    The binding packages the raw matrix entry and the normalized adapter
    contract into one object so later runners do not need to reparsing YAML.
    """

    model_name: str
    adapter_name: str
    contract: AdapterContract
    eligible_long_term: bool
    eligible_m4: bool
    review_status: str
    eligibility_source: str
    eligibility_reason: str
    supports_regular_mts: bool | None
    supports_univariate: bool | None
    notes: tuple[str, ...]
    raw_capability: dict[str, Any]

    def is_eligible_for(self, task_name: TaskName) -> bool:
        """Return whether the model enters the automated ranking for a task.

        Args:
            task_name: Benchmark task to inspect.

        Returns:
            ``True`` when the model is eligible for that task.
        """
        if task_name == "long_term":
            return self.eligible_long_term
        if task_name == "m4":
            return self.eligible_m4
        raise KeyError(f"Unsupported task '{task_name}'.")