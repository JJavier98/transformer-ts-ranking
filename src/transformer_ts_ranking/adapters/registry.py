"""Static adapter-family registry for the benchmark runner.

The registry is declarative: it does not build actual batches yet, but it
formalizes the expected keys for each adapter family and benchmark task.
"""

from __future__ import annotations

from .base import AdapterContract, AdapterTaskContract

_ENCODER_ONLY = AdapterContract(
    family_name="encoder_only",
    description="Regular forecasting models that consume an encoder history and predict directly.",
    task_contracts={
        "long_term": AdapterTaskContract(
            task_name="long_term",
            train_batch_keys=("x", "y"),
            eval_batch_keys=("x", "y"),
            predict_batch_keys=("x",),
            optional_batch_keys=("x_mark",),
        ),
        "m4": AdapterTaskContract(
            task_name="m4",
            train_batch_keys=("x", "y"),
            eval_batch_keys=("x", "y"),
            predict_batch_keys=("x",),
            optional_batch_keys=("x_mark",),
        ),
    },
)

_SEQ2SEQ = AdapterContract(
    family_name="seq2seq",
    description="Encoder-decoder models that require temporal marks and decoder context.",
    task_contracts={
        "long_term": AdapterTaskContract(
            task_name="long_term",
            train_batch_keys=("x", "x_mark", "y", "y_mark"),
            eval_batch_keys=("x", "x_mark", "y", "y_mark"),
            predict_batch_keys=("x", "x_mark", "y_mark"),
            optional_batch_keys=("y_full",),
        ),
        "m4": AdapterTaskContract(
            task_name="m4",
            train_batch_keys=("x", "x_mark", "y", "y_mark"),
            eval_batch_keys=("x", "x_mark", "y", "y_mark"),
            predict_batch_keys=("x", "x_mark", "y_mark"),
            optional_batch_keys=("y_full",),
        ),
    },
)

_EXOGENOUS_AWARE = AdapterContract(
    family_name="exogenous_aware",
    description="Models that follow a time-marked forecasting interface and may consume known future covariates.",
    task_contracts={
        "long_term": AdapterTaskContract(
            task_name="long_term",
            train_batch_keys=("x", "x_mark", "y", "y_mark"),
            eval_batch_keys=("x", "x_mark", "y", "y_mark"),
            predict_batch_keys=("x", "x_mark", "y_mark"),
            optional_batch_keys=("known_covariates", "future_covariates", "y_full"),
        ),
        "m4": AdapterTaskContract(
            task_name="m4",
            train_batch_keys=("x", "x_mark", "y", "y_mark"),
            eval_batch_keys=("x", "x_mark", "y", "y_mark"),
            predict_batch_keys=("x", "x_mark", "y_mark"),
            optional_batch_keys=("known_covariates", "future_covariates", "y_full"),
        ),
    },
)

_SPATIAL = AdapterContract(
    family_name="spatial",
    description="Models that require an additional spatial view or graph-aware context beyond plain regular series.",
    task_contracts={
        "long_term": AdapterTaskContract(
            task_name="long_term",
            train_batch_keys=("x", "y"),
            eval_batch_keys=("x", "y"),
            predict_batch_keys=("x",),
            optional_batch_keys=("x_mark", "spatial_context", "adjacency_matrix"),
            notes="Spatial families are only valid when the benchmark protocol defines the required spatial context.",
        ),
        "m4": AdapterTaskContract(
            task_name="m4",
            train_batch_keys=("x", "y"),
            eval_batch_keys=("x", "y"),
            predict_batch_keys=("x",),
            optional_batch_keys=("x_mark", "spatial_context", "adjacency_matrix"),
            notes="Spatial families are only valid when the benchmark protocol defines the required spatial context.",
        ),
    },
)

_IRREGULAR = AdapterContract(
    family_name="irregular",
    description="Models that operate on irregular observations with explicit timestamps and masks.",
    task_contracts={
        "long_term": AdapterTaskContract(
            task_name="long_term",
            train_batch_keys=("x", "x_time", "x_mask", "pred_time", "y"),
            eval_batch_keys=("x", "x_time", "x_mask", "pred_time", "y"),
            predict_batch_keys=("x", "x_time", "x_mask", "pred_time"),
            notes="The regular long-term benchmark does not generate this irregular input layout today.",
        ),
        "m4": AdapterTaskContract(
            task_name="m4",
            train_batch_keys=("x", "x_time", "x_mask", "pred_time", "y"),
            eval_batch_keys=("x", "x_time", "x_mask", "pred_time", "y"),
            predict_batch_keys=("x", "x_time", "x_mask", "pred_time"),
            notes="The current M4 point-forecast protocol does not generate this irregular input layout today.",
        ),
    },
)


ADAPTER_CONTRACTS = {
    "encoder_only": _ENCODER_ONLY,
    "seq2seq": _SEQ2SEQ,
    "exogenous_aware": _EXOGENOUS_AWARE,
    "spatial": _SPATIAL,
    "irregular": _IRREGULAR,
}


def get_adapter_contract(adapter_name: str) -> AdapterContract:
    """Resolve one adapter family from the static registry.

    Args:
        adapter_name: Family identifier stored in the capability matrix.

    Returns:
        The registered adapter contract.
    """
    if adapter_name not in ADAPTER_CONTRACTS:
        available = ", ".join(sorted(ADAPTER_CONTRACTS))
        raise KeyError(f"Unknown adapter family '{adapter_name}'. Available: {available}")
    return ADAPTER_CONTRACTS[adapter_name]


def list_adapter_contracts() -> list[AdapterContract]:
    """Return the registered adapter contracts in stable name order.

    Returns:
        All known adapter contracts sorted by family name.
    """
    return [ADAPTER_CONTRACTS[name] for name in sorted(ADAPTER_CONTRACTS)]