"""Benchmark-safe model configuration builders.

Every eligible model needs a config dict that is compatible with the
library's create_model() factory.  This module centralises all model-
specific overrides so neither the runner nor the engine contains any
``if model_name ==`` branches.

CLAUDE.md documented overrides are applied here:
  - fedformer  : BERT-large default is reduced to avoid 8 GB OOM.
  - patchtst   : Default d_model=768 is tight; benchmarks use d_model=256.
  - lag_llama  : use_pinball_loss forced off for deterministic point forecasts.
  - earthformer: data_mode forced to '1d' to prevent accidental 3D mode.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Per-model hard-cap overrides applied on top of the canonical base config.
# These exist solely to keep all models within the 8 GB VRAM budget and to
# prevent known behavioural anomalies.
# ---------------------------------------------------------------------------
_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "fedformer": {
        # Default BERT-large scale would OOM on 8 GB GPU at batch_size ≥ 4.
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "d_model": 256,
        "d_ff": 512,
        "n_heads": 4,
    },
    "patchtst": {
        # Default d_model=768 is tight; reduce for reproducible benchmarks.
        "d_model": 256,
        "n_heads": 8,
        "d_ff": 512,
    },
    "lag_llama": {
        # Disable probabilistic pinball loss; benchmark uses deterministic MSE.
        "use_pinball_loss": False,
    },
    "earthformer": {
        # Prevent accidental 3D image mode.
        "data_mode": "1d",
    },
}

# Models whose seq2seq nature requires an explicit label_len in the config.
# All other eligible models receive label_len=0 (encoder-only style).
_SEQ2SEQ_MODELS = frozenset({
    "autoformer",
    "basisformer",
    "cats",
    "deformable_tst",
    "fedformer",
    "informer",
    "nonstationary_transformer",
    "patchtst",
    "pyraformer",
    "quatformer",
    "reformer",
    "scaleformer",
    "transformer",
})


def filter_config_for_model(model_name: str, config: dict[str, Any]) -> Any:
    """Build a model-specific config object (or filtered dict) from a superset dict.

    Each model's ``__init__`` may either accept a plain dict and convert it
    internally (e.g. ``iTransformerConfig(**dict)``), or it may require a
    typed config dataclass instance directly.  This helper:
      1. Resolves the model's registered config class.
      2. Filters the input dict to only the declared dataclass fields.
      3. Instantiates and returns the config class so every model receives
         the type it expects.

    Args:
        model_name: Canonical model key.
        config: Candidate config dict (possibly with superset of keys).

    Returns:
        Instantiated config class instance, or the filtered dict as fallback.
    """
    import dataclasses
    import sys
    from pathlib import Path

    lib_root = str(Path(__file__).resolve().parents[5] / "s-transformers-lib")
    if lib_root not in sys.path:
        sys.path.insert(0, lib_root)

    try:
        from src.models import get_config_class  # noqa: PLC0415
        config_cls = get_config_class(model_name)
        if dataclasses.is_dataclass(config_cls):
            valid_fields = {f.name for f in dataclasses.fields(config_cls)}
            filtered = {k: v for k, v in config.items() if k in valid_fields}
            return config_cls(**filtered)
    except Exception:
        pass  # Fall back to passing the full dict; let the model handle it.
    return config


def build_long_term_config(
    model_name: str,
    n_channels: int,
    pred_len: int,
    seq_len: int,
    label_len: int,
    freq: str,
) -> dict[str, Any]:
    """Build a benchmark-safe config dict for one model × dataset × horizon.

    The config satisfies the minimal canonical contract expected by every
    eligible model and applies the CLAUDE.md-documented safety overrides.

    Args:
        model_name: Canonical model key (e.g. ``"itransformer"``).
        n_channels: Number of input/output channels from the dataset.
        pred_len: Forecast horizon for this benchmark run.
        seq_len: Encoder input length.
        label_len: Decoder label-context length (from dataset manifest).
        freq: Pandas-style frequency code (e.g. ``"h"``, ``"t"``).

    Returns:
        Config dict ready for ``create_model(model_name, config=...)``.
    """
    effective_label_len = label_len if model_name in _SEQ2SEQ_MODELS else 0

    config: dict[str, Any] = {
        # Core forecasting dimensions
        "seq_len": seq_len,
        "pred_len": pred_len,
        "label_len": effective_label_len,
        # Channel dimensions (models use different names for the same concept)
        "enc_in": n_channels,
        "dec_in": n_channels,
        "c_out": n_channels,
        "num_features": n_channels,
        "input_size": n_channels,
        # Frequency hint used by time-feature embeddings
        "freq": freq,
        # Conservative defaults to stay within 8 GB VRAM
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.1,
    }

    config.update(_MODEL_OVERRIDES.get(model_name, {}))
    return config


def build_m4_config(
    model_name: str,
    seq_len: int,
    horizon: int,
) -> dict[str, Any]:
    """Build a config dict for M4 short-term forecasting (univariate).

    Args:
        model_name: Canonical model key.
        seq_len: Encoder input length.
        horizon: M4 forecast horizon for the frequency slice.

    Returns:
        Config dict ready for ``create_model(model_name, config=...)``.
    """
    effective_label_len = min(horizon // 2, 24) if model_name in _SEQ2SEQ_MODELS else 0

    config: dict[str, Any] = {
        "seq_len": seq_len,
        "pred_len": horizon,
        "label_len": effective_label_len,
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1,
        "num_features": 1,
        "input_size": 1,
        "freq": "t",
        "d_model": 64,
        "d_ff": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.1,
    }

    config.update(_MODEL_OVERRIDES.get(model_name, {}))
    return config
