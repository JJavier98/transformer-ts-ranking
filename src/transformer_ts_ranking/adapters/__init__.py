"""Adapter contracts and capability-matrix resolution helpers."""

from .base import AdapterContract, AdapterTaskContract, ModelAdapterBinding
from .registry import get_adapter_contract, list_adapter_contracts
from .resolution import iter_task_model_bindings, resolve_model_adapter

__all__ = [
    "AdapterContract",
    "AdapterTaskContract",
    "ModelAdapterBinding",
    "get_adapter_contract",
    "iter_task_model_bindings",
    "list_adapter_contracts",
    "resolve_model_adapter",
]