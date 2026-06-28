"""Benchmark-safe model configuration builders.

Every eligible model needs a config dict that is compatible with the
library's create_model() factory.  This module centralises all model-
specific overrides so neither the runner nor the engine contains any
``if model_name ==`` branches.

CLAUDE.md documented overrides are applied here:
  - fedformer           : BERT-large default is reduced to avoid 8 GB OOM.
  - patchtst            : Default d_model=768 is tight; benchmarks use d_model=256.
  - lag_llama           : use_pinball_loss forced off for deterministic point forecasts.
  - earthformer         : data_mode forced to '1d' to prevent accidental 3D mode.
  - spacetimeformer     : max_seq_len raised to 1024 (default 512 causes CUDA index
                          out-of-bounds for h=720: decoder length = label_len+pred_len = 768).
  - lag_llama_pretrained: d_model forced to 144 to match pretrained checkpoint dims.
  - etsformer           : d_layers forced equal to e_layers (model assertion requirement).

Frequency note — ``freq`` in all configs is fixed to ``'h'`` (4 time features).
The long-term and M4 loaders both produce exactly 4 time features regardless of
dataset frequency, matching FREQ_MAP['h']=4 used by TimeFeatureEmbedding.  Passing
the actual dataset frequency (e.g. 'w'→2, 't'→5) would cause a shape mismatch.

Runtime patches (``_MODEL_PATCHES`` / ``patch_model_instance``):
  Bug fixes that cannot live in the library source are applied as instance-level
  monkey-patches right after ``create_model()`` returns.  The dispatch dict
  ``_MODEL_PATCHES`` maps model_name → callable(model) so neither the engine nor
  the runner contains any ``if model_name ==`` branches.
"""

from __future__ import annotations

import types
from typing import Any


# ---------------------------------------------------------------------------
# Per-model hard-cap overrides applied on top of the canonical base config.
# These exist solely to keep all models within the 8 GB VRAM budget and to
# prevent known behavioural anomalies.
# ---------------------------------------------------------------------------
_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "chronos_bolt": {
        # Official pretrained Chronos-Bolt from HuggingFace (zero-shot).
        # chronos-bolt-small: ~21 M params, recommended for benchmark balance.
        # Switch to "amazon/chronos-bolt-base" (~205 M) for best accuracy.
        "model_id": "amazon/chronos-bolt-small",
        "torch_dtype": "float32",
    },
    "lag_llama_pretrained": {
        # Official Lag-Llama pretrained weights from HuggingFace (zero-shot).
        # ALL architecture params must be overridden explicitly because
        # build_{long_term,m4}_config injects generic defaults (d_ff=256,
        # n_heads=4) that would be passed through filter_config_for_model and
        # override the dataclass field defaults, causing a state_dict size
        # mismatch.  Checkpoint shapes (from model.safetensors inspection):
        #   embed_inputs.weight : (144, 92)  → d_model=144, input_dim=92
        #   mlp.gate_proj.weight: (512, 144) → d_ff=512
        #   self_attn.q_proj    : (144, 144) → n_heads flexible but 8 is correct
        "hf_repo": "time-series-foundation-models/Lag-Llama",
        "hf_filename": "model.safetensors",
        "scaling": "mean",
        "d_model": 144,
        "d_ff": 512,
        "n_heads": 8,
        "n_layers": 8,
    },
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
    "spacetimeformer": {
        # Default max_seq_len=512 is too small for h=720: the absolute positional
        # embedding table is indexed up to label_len+pred_len=768 on the decoder,
        # causing a CUDA device-side assert (index out-of-bounds).
        "max_seq_len": 1024,
    },
    "etsformer": {
        # ETSFormer asserts e_layers == d_layers.  The base config sets d_layers=1
        # (encoder-only default) which violates this constraint.
        "d_layers": 2,
    },
    "scaleformer": {
        # Default scales=[8,4,2,1] requires pred_len % coarsest_scale == 0.
        # Long-term illness horizons [24, 36, 48, 60] violate this for scales
        # containing 8 (36 % 8 != 0, 60 % 8 != 0).  scales=[1] disables
        # multi-scale but makes the model compatible with any pred_len across
        # all benchmark datasets (long-term and M4).
        "scales": [1],
    },
}

# M4-specific overrides applied ON TOP of _MODEL_OVERRIDES for M4 configs only.
# These resolve structural incompatibilities between model defaults and the small
# pred_len values used in some M4 frequency slices (Yearly=6, Weekly=13).
_M4_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "multipatchformer": {
        # Default n_sar_steps=8 requires pred_len >= 8.  M4 Yearly (pred_len=6)
        # violates this.  Setting n_sar_steps=4 satisfies all M4 horizons (≥ 6).
        "n_sar_steps": 4,
    },
}

# ---------------------------------------------------------------------------
# Models that do NOT support bfloat16 in PyTorch autocast.
# These models use operations (FFT, stochastic distributions, custom CUDA
# kernels, or legacy integer indices) that raise TypeError/RuntimeError when
# run in bf16 mode.  The engine's _autocast_ctx() skips autocast for these
# models and runs fp32.
#
# Confirmed failures on A100 with torch.amp.autocast(bf16) and
# PyTorch 2.5.1+cu124, all raising TypeError: Got unsupported ScalarType
# BFloat16 (or RuntimeError for FFT models):
#
#   Group 1 — FFT operations (torch.fft.rfft does not support bf16):
#     autoformer    — AutoCorrelation layer via torch.fft.rfft
#     etsformer     — ExponentialSmoothing via torch.fft
#     fedformer     — FEDformer spectral mixing via torch.fft
#     pathformer    — patch-wise FFT in PatchEncoder
#
#   Group 2 — Stochastic / distribution ops (require float32 precision):
#     airformer     — Normal distribution log_prob in stochastic encoder
#
#   Group 3 — Custom attention ops incompatible with bf16 kernels:
#     crossformer   — cross-segment router attention internal op
#     earthformer   — cuboid self-attention internal op
#     informer      — ProbSparse attention (topk + cumsum path)
#     pyraformer    — CSCM pyramidal convolution + local attention
#     reformer      — LSH (Locality-Sensitive Hashing) attention bucket ops
#     spacetimeformer — factorised space-time attention (multi-head path)
#     triformer     — triangular attention patch ops
#
#   Group 4 — Encoder-decoder models with non-bf16-compatible decoder inputs:
#     transformer   — vanilla enc-dec: decoder cross-attention raises BFloat16
#     tft           — Temporal Fusion Transformer: GRN / LSTM gating path
#
# Note: pyraformer, transformer, tft, spacetimeformer also belong to Group B
# (require x_mark_dec / x_dec via ForecastInput) — the dtype mismatch between
# fp32 time-mark inputs and bf16 activations may be a contributing factor.
# ---------------------------------------------------------------------------
_MODEL_NO_BF16: frozenset[str] = frozenset({
    # Group 1 — FFT
    "airformer",
    "autoformer",
    "crossformer",
    "earthformer",
    "etsformer",
    "fedformer",
    "pathformer",
    # Group 3 — custom attention ops
    "informer",
    "pyraformer",
    "reformer",
    "spacetimeformer",
    "triformer",
    # Group 4 — encoder-decoder with bf16-incompatible decoder path
    "transformer",
    "tft",
})


def is_bf16_safe(model_name: str) -> bool:
    """Return True if the model can run inside torch.amp.autocast(bf16).

    Models that use FFT, stochastic distributions, or custom CUDA kernels
    that are not bf16-compatible are excluded.  All others can benefit from
    bf16 tensor-core throughput on A100/H100 GPUs.

    Args:
        model_name: Canonical model key.

    Returns:
        ``False`` for models known to fail with bf16 autocast; ``True`` otherwise.
    """
    return model_name not in _MODEL_NO_BF16


# ---------------------------------------------------------------------------
# Per-model batch-size overrides.
# Values are either a flat int (all horizons) or a {horizon: batch_size} dict.
# Resolved by get_batch_size_override(); passed to the engine per run so the
# runner-level batch_size is not changed globally.
# ---------------------------------------------------------------------------
_MODEL_BATCH_OVERRIDES: dict[str, "int | dict[int, int]"] = {
    "spacetimeformer": 4,
    # Decoder attention has TWO quadratic terms:
    #   temporal: B × C × H × T_dec² × 4 bytes
    #   spatial:  B × T_dec × H × C² × 4 bytes
    # At default B=16 these exceed 40 GB for:
    #   - any dataset at h=720 (T_dec=768, even C=7: 8.5 GB temporal)
    #   - electricity (C=321) at h≥96 (temporal: 3.4 GB, spatial: 7.6 GB at h=96)
    #   - traffic (C=862) at all horizons (spatial alone ~55 GB at h=96, B=16)
    # B=4 keeps small-channel datasets (C≤21) under 3.5 GB total at all horizons.
    # For electricity it rescues h=96 (5.5 GB), h=192 (11.1 GB), h=336 (22.2 GB);
    # h=720 still OOMs (68.7 GB).  Traffic OOMs at all horizons even at B=1.
    # Those combos produce error rows in the parquet and are excluded from the paper.
}

# ---------------------------------------------------------------------------
# Per-model encoder context-length overrides.
# When non-None, the engine truncates the encoder input x (and x_mark) to
# the last context_len time steps before every fit() and predict() call.
# Required for models whose attention is O(T²) in the encoder sequence.
# ---------------------------------------------------------------------------
_MODEL_CONTEXT_LEN: dict[str, int] = {
    "contiformer": 96,
    # ODELinear.forward() allocates [B, T, T, D] pairwise integral tensors.
    # At T=336, B=16: ~11 GB.  Truncating to T=96 reduces this to ~150 MB
    # while preserving the continuous-time ODE semantics — the model can operate
    # on any context length; 96 is the benchmark's chosen budget.
    # For M4 (seq_len=96 by default) truncation is a no-op.
}


def get_batch_size_override(
    model_name: str,
    pred_len: int,
    default: int,
) -> int:
    """Return the effective batch size for a model × horizon combination.

    Models that exceed VRAM at the global batch size register a per-horizon
    (or flat) override in ``_MODEL_BATCH_OVERRIDES``.

    Args:
        model_name: Canonical model key.
        pred_len: Forecast horizon for the current run.
        default: Runner-level default batch size.

    Returns:
        Effective batch size to pass to the engine for this run.
    """
    override = _MODEL_BATCH_OVERRIDES.get(model_name)
    if override is None:
        return default
    if isinstance(override, int):
        return override
    return override.get(pred_len, default)


def get_context_len_override(model_name: str) -> "int | None":
    """Return the encoder context-length override for a model, or None.

    When non-None the engine truncates the encoder input to the last
    ``context_len`` time steps before every ``fit()`` and ``predict()`` call.

    Args:
        model_name: Canonical model key.

    Returns:
        Context length in time steps, or ``None`` (no truncation).
    """
    return _MODEL_CONTEXT_LEN.get(model_name)

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
    "spacetimeformer",
    "tft",
    # TFT uses label_len in 10 places (batch_y[:, :self.label_len, :],
    # x_mark_dec[:, c.label_len:, :], etc.).  Without this entry label_len=0
    # and Python's x[:, -0:, :] returns the full encoder sequence (since -0==0),
    # feeding the wrong decoder context and silently corrupting TFT outputs.
    "transformer",
})


def filter_config_for_model(model_name: str, config: dict[str, Any]) -> Any:
    """Build a model-specific config object (or filtered dict) from a superset dict.

    Each model's ``__init__`` may either accept a plain dict and convert it
    internally (e.g. ``iTransformerConfig(**dict)``), or it may require a
    typed config dataclass instance directly.  This helper:
      1. Resolves the model's registered config class.
      2. Filters the input dict to the parameters accepted by ``__init__``.
      3. Instantiates and returns the config class so every model receives
         the type it expects.

    We use ``inspect.signature`` rather than ``dataclasses.fields`` because
    several config classes (AirFormerConfig, CardConfig, CrossformerConfig,
    EarthformerConfig, ContiFormerConfig, …) inherit from the ForecastBaseConfig
    dataclass but declare their own parameters (seq_len, pred_len, enc_in, …)
    via a regular ``__init__``.  ``dataclasses.fields`` only sees the five
    parent fields and silently drops the architecture parameters, causing every
    run to use the default pred_len=96 regardless of the actual horizon.

    Args:
        model_name: Canonical model key.
        config: Candidate config dict (possibly with superset of keys).

    Returns:
        Instantiated config class instance, or the filtered dict as fallback.
    """
    import inspect
    import sys
    from pathlib import Path

    lib_root = str(Path(__file__).resolve().parents[5] / "s-transformers-lib")
    if lib_root not in sys.path:
        sys.path.insert(0, lib_root)

    try:
        from src.models import get_config_class  # noqa: PLC0415
        config_cls = get_config_class(model_name)
        sig = inspect.signature(config_cls.__init__)
        # Exclude 'self' and variadic **kwargs / *args so they don't consume
        # the whole dict and mask missing required parameters.
        valid_params = {
            k for k, p in sig.parameters.items()
            if k not in ("self",)
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        filtered = {k: v for k, v in config.items() if k in valid_params}
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
        # Always 'h' (FREQ_MAP['h']=4) regardless of dataset frequency.
        # The long-term loader produces exactly 4 time features for every dataset
        # (month, day, weekday, hour — hour=0 for sub-daily frequencies).
        # Passing the actual freq (e.g. 'w'→2, 't'→5) causes a shape mismatch
        # in models that build TimeFeatureEmbedding with FREQ_MAP[freq] input dims.
        "freq": "h",
        # Conservative defaults to stay within 8 GB VRAM
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.1,
    }

    config.update(_MODEL_OVERRIDES.get(model_name, {}))
    return config  # _M4_MODEL_OVERRIDES not applied here — long_term only


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
        # M4 window dataset always produces 4-zero time features; use 'h' so
        # TimeFeatureEmbedding builds a Linear(4, d_model) rather than (5, d_model).
        "freq": "h",
        "d_model": 64,
        "d_ff": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "dropout": 0.1,
    }

    config.update(_MODEL_OVERRIDES.get(model_name, {}))
    config.update(_M4_MODEL_OVERRIDES.get(model_name, {}))
    return config


# ---------------------------------------------------------------------------
# Runtime model-instance patches.
#
# Some library bugs cannot be fixed by config overrides — they live inside
# private methods of the model class.  Since the library is a read-only
# submodule we apply instance-level monkey-patches immediately after
# ``create_model()`` returns, before the model is used for training or
# inference.
#
# Each entry maps a model_name → callable(model_instance) that modifies the
# instance in-place.  The dispatch is purely data-driven so neither the
# engine nor the runner contains ``if model_name ==`` branches.
# ---------------------------------------------------------------------------

def _patch_lag_llama_pretrained(model: Any) -> None:
    """Fix _build_lag_features: loc/scale arrive as (B, 1), not (B, 1, 1).

    The library calls ``loc.expand(B, T, 1)`` on a 2-D tensor.  PyTorch
    prepends a singleton when the target rank is higher, turning (B, 1)
    into (1, B, 1), then fails to expand dim-1 from B to T because B != T
    (error: "expanded size (T) must match existing size (B)").

    Fix: insert ``unsqueeze(1)`` so the tensor is (B, 1, 1) before the
    expand, which then correctly yields (B, T, 1).

    This is an instance-level monkey-patch — the source file is not modified.
    """
    import torch as _torch

    def _fixed_build_lag_features(
        self: Any,
        ctx: "_torch.Tensor",
        loc: "_torch.Tensor",
        scale: "_torch.Tensor",
    ) -> "_torch.Tensor":
        B, T = ctx.shape
        dev = ctx.device
        lag_features = []
        for lag in self._lags_seq:
            if lag >= T:
                lag_features.append(_torch.zeros(B, T, device=dev, dtype=ctx.dtype))
            else:
                padded = _torch.cat(
                    [_torch.zeros(B, lag, device=dev, dtype=ctx.dtype), ctx[:, :T - lag]],
                    dim=1,
                )
                lag_features.append(padded)
        lag_tensor = _torch.stack(lag_features, dim=-1)          # (B, T, 90)
        # unsqueeze(1): (B,1) → (B,1,1) so expand(B,T,1) works correctly.
        log_loc   = _torch.log(loc.abs() + 1.0).unsqueeze(1).expand(B, T, 1)
        log_scale = _torch.log(scale + 1e-8).unsqueeze(1).expand(B, T, 1)
        return _torch.cat([lag_tensor, log_loc, log_scale], dim=-1)  # (B, T, 92)

    model._build_lag_features = types.MethodType(_fixed_build_lag_features, model)


# Map model_name → patch callable.  Applied by patch_model_instance() in the engine.
_MODEL_PATCHES: dict[str, Any] = {
    "lag_llama_pretrained": _patch_lag_llama_pretrained,
}


def patch_model_instance(model_name: str, model: Any) -> None:
    """Apply known runtime patches to a freshly created model instance.

    Called by the engine immediately after ``create_model()`` so every run
    benefits from the fix without any ``if model_name ==`` in the engine.

    Args:
        model_name: Canonical model key.
        model: Model instance returned by ``create_model()``.
    """
    patcher = _MODEL_PATCHES.get(model_name)
    if patcher is not None:
        patcher(model)
