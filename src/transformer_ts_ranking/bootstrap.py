"""Bootstrap versioned benchmark manifests from the static audit.

This module turns the raw audit snapshot into durable benchmark configuration.
It is the bridge between discovery and execution: once these manifests exist,
the rest of the pipeline can operate without re-inspecting the external library.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .configuration import write_yaml
from .discovery.model_audit import audit_model_library, write_audit_artifacts

__all__ = ["materialize_bootstrap_manifests"]


DEFAULT_LONG_TERM_DATASETS = {
    "manifest_kind": "long_term_datasets",
    "curation_status": "bootstrap",
    "defaults": {
        "date_column": "date",
        "target_columns": ["OT"],
        "feature_mode": "multivariate",
        "scaler": "standard",
        "split_strategy": "ratio",
        "split": {"train_ratio": 0.7, "val_ratio": 0.1},
        "seq_len": 96,
        "label_len": 48,
        "stride": 1,
    },
    "datasets": {
        "electricity": {
            "relative_path": "data/long_term_forecast/electricity/electricity.csv",
            "frequency": "h",
            "horizons": [96, 192, 336, 720],
        },
        "exchange_rate": {
            "relative_path": "data/long_term_forecast/exchange_rate/exchange_rate.csv",
            "frequency": "d",
            "horizons": [96, 192, 336, 720],
        },
        "illness": {
            "relative_path": "data/long_term_forecast/illness/national_illness.csv",
            "frequency": "w",
            "horizons": [24, 36, 48, 60],
        },
        "traffic": {
            "relative_path": "data/long_term_forecast/traffic/traffic.csv",
            "frequency": "h",
            "horizons": [96, 192, 336, 720],
        },
        "weather": {
            "relative_path": "data/long_term_forecast/weather/weather.csv",
            "frequency": "t",
            "horizons": [96, 192, 336, 720],
        },
        "ETTh1": {
            "relative_path": "data/long_term_forecast/ETT-small/ETTh1.csv",
            "frequency": "h",
            "horizons": [96, 192, 336, 720],
            "split_strategy": "fixed_counts",
            "split": {"train": 8640, "val": 2880, "test": 2880},
        },
        "ETTh2": {
            "relative_path": "data/long_term_forecast/ETT-small/ETTh2.csv",
            "frequency": "h",
            "horizons": [96, 192, 336, 720],
            "split_strategy": "fixed_counts",
            "split": {"train": 8640, "val": 2880, "test": 2880},
        },
        "ETTm1": {
            "relative_path": "data/long_term_forecast/ETT-small/ETTm1.csv",
            "frequency": "t",
            "horizons": [96, 192, 336, 720],
            "split_strategy": "fixed_counts",
            "split": {"train": 34560, "val": 11520, "test": 11520},
        },
        "ETTm2": {
            "relative_path": "data/long_term_forecast/ETT-small/ETTm2.csv",
            "frequency": "t",
            "horizons": [96, 192, 336, 720],
            "split_strategy": "fixed_counts",
            "split": {"train": 34560, "val": 11520, "test": 11520},
        },
    },
}


DEFAULT_M4_DATASETS = {
    "manifest_kind": "m4_datasets",
    "curation_status": "bootstrap",
    "defaults": {
        "relative_root": "data/short_term_forecast/m4",
        "info_file": "M4-info.csv",
        "naive2_file": "submission-Naive2.csv",
    },
    "frequencies": {
        "Yearly": {
            "train_file": "Yearly-train.csv",
            "test_file": "Yearly-test.csv",
        },
        "Quarterly": {
            "train_file": "Quarterly-train.csv",
            "test_file": "Quarterly-test.csv",
        },
        "Monthly": {
            "train_file": "Monthly-train.csv",
            "test_file": "Monthly-test.csv",
        },
        "Weekly": {
            "train_file": "Weekly-train.csv",
            "test_file": "Weekly-test.csv",
        },
        "Daily": {
            "train_file": "Daily-train.csv",
            "test_file": "Daily-test.csv",
        },
        "Hourly": {
            "train_file": "Hourly-train.csv",
            "test_file": "Hourly-test.csv",
        },
    },
}


DEFAULT_TRAINING_PRESETS = {
    "manifest_kind": "training_presets",
    "curation_status": "bootstrap",
    "presets": {
        "smoke": {
            "seeds": [42],
            "max_models": 3,
            "max_horizons_per_dataset": 1,
            "representative_models": ["patchtst", "basisformer", "fedformer"],
            "batch_size": 16,
            "max_epochs": 1,
            "dry_run": True,
        },
        "standard": {
            "seeds": [42, 123, 2026],
            "max_models": None,
            "max_horizons_per_dataset": None,
            "batch_size": 32,
            "max_epochs": 20,
            "early_stopping_patience": 5,
            "dry_run": False,
        },
        "paper_ready": {
            "seeds": [42, 123, 2026],
            "max_models": None,
            "max_horizons_per_dataset": None,
            "batch_size": 32,
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "dry_run": False,
        },
    },
}


ELIGIBILITY_POLICY = {
    "eligible_long_term": "Controls whether a model enters the automated long-term leaderboard.",
    "eligible_m4": "Controls whether a model enters the automated M4 leaderboard.",
    "both_false_behavior": (
        "If both flags are false, the model is excluded from the main automated rankings "
        "and should appear in the coverage table until it is reviewed or adapted."
    ),
    "source_precedence": ["manual_override", "bootstrap_heuristic"],
}


MANUAL_CAPABILITY_OVERRIDES = {
    "airformer": {
        "supports_regular_mts": True,
        "supports_univariate": True,
        "requires_spatial_structure": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "adapter_name": "encoder_only",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed AirFormer executes as an encoder-only forecaster on both "
            "regular multivariate long-term data and univariate M4-style inputs without a required "
            "external spatial graph."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe succeeded for long_term and M4 with the encoder-only profile.",
        ],
    },
    "chronos2": {
        "requires_exogenous": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "adapter_name": "encoder_only",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Chronos2Config defaults n_future_covariates=0 and Chronos2.forward() accepts x_enc "
            "with optional time marks and optional future covariates, so the model should enter "
            "both benchmark tracks by default."
        ),
        "notes_append": [
            "Manual override: Chronos2 does not require exogenous covariates by default; bootstrap exclusion removed.",
        ],
    },
    "contiformer": {
        "supports_regular_mts": True,
        "supports_univariate": True,
        "requires_irregular_times": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "adapter_name": "encoder_only",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed ContiFormer can execute on both benchmark tracks using its regular-time "
            "fallback, so it should be included in the automated rankings."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe succeeded for long_term and M4 with the encoder-only profile.",
        ],
    },
    "earthformer": {
        "supports_regular_mts": True,
        "supports_univariate": True,
        "requires_spatial_structure": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "adapter_name": "encoder_only",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed Earthformer runs in documented 1d mode for both long-term and M4-style "
            "series, so it should not be excluded as spatial-only."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe succeeded for long_term and M4 with Earthformer configured in 1d mode.",
        ],
    },
    "spacetimeformer": {
        "supports_regular_mts": True,
        "supports_univariate": True,
        "requires_spatial_structure": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "adapter_name": "seq2seq",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed Spacetimeformer executes on both benchmark tracks with its standard seq2seq "
            "interface, so it should not be excluded as requiring external spatial structure."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe succeeded for long_term and M4 with the seq2seq profile.",
        ],
    },
    "tft": {
        "supports_univariate": True,
        "requires_exogenous": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "adapter_name": "seq2seq",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed TFT works on both benchmark tracks with time marks but without mandatory "
            "external exogenous covariates, so the bootstrap exclusion was too strict."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe succeeded for long_term and M4 with the seq2seq profile.",
        ],
    },
    "timexer": {
        "supports_univariate": True,
        "requires_exogenous": False,
        "eligible_long_term": True,
        "eligible_m4": True,
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed TimeXer executes on both benchmark tracks with time-mark inputs and does not "
            "require separate exogenous covariates to participate in the automated rankings."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe succeeded for long_term and M4 with the time-marked encoder profile.",
        ],
    },
    "tpatchgnn": {
        "supports_univariate": False,
        "eligible_long_term": False,
        "eligible_m4": False,
        "adapter_name": "irregular",
        "review_status": "manual_override",
        "eligibility_source": "manual_override",
        "eligibility_reason": (
            "Runtime probe confirmed tPatchGNN expects irregular patched inputs with explicit observation times and "
            "masks, so it remains outside the regular long-term and M4 automated rankings."
        ),
        "notes_append": [
            "Manual override: runtime compatibility probe kept tPatchGNN excluded because the regular benchmark tasks do not provide its required irregular patched input format.",
        ],
    },
}


def _bootstrap_eligibility_reason(model: dict) -> str:
    """Explain the current bootstrap eligibility decision for one model.

    Args:
        model: Capability entry emitted by the static audit.

    Returns:
        A human-readable explanation for the current bootstrap decision.
    """
    if not model.get("registered", False):
        return "Excluded because the model is missing from the public registry."
    if model.get("requires_spatial_structure"):
        return "Excluded because the bootstrap audit classifies the model as requiring spatial structure."
    if model.get("requires_irregular_times"):
        return "Excluded because the bootstrap audit classifies the model as requiring irregular timestamps."
    if model.get("requires_exogenous"):
        return "Excluded because the bootstrap audit classifies the model as requiring exogenous covariates."
    if not model.get("supports_regular_mts"):
        return "Excluded because the bootstrap audit could not validate regular multivariate support yet."
    return "Included by the bootstrap audit for comparable benchmark tracks."


def _apply_manual_overrides(versioned_payload: dict) -> dict:
    """Apply curated capability overrides on top of the bootstrap matrix.

    Args:
        versioned_payload: Bootstrap capability matrix before manual review.

    Returns:
        The same payload enriched with manual overrides and appended notes.
    """
    for model in versioned_payload.get("models", []):
        model.setdefault("eligibility_source", "bootstrap_heuristic")
        model.setdefault("eligibility_reason", _bootstrap_eligibility_reason(model))

        override = MANUAL_CAPABILITY_OVERRIDES.get(model["model_name"])
        if override is None:
            continue

        notes = list(model.get("notes", []))
        for note in override.get("notes_append", []):
            if note not in notes:
                notes.append(note)
        # Notes are appended rather than overwritten so the bootstrap evidence is
        # still visible after the manual review step.
        model.update({key: value for key, value in override.items() if key != "notes_append"})
        model["notes"] = notes

    return versioned_payload


def _refresh_summary(versioned_payload: dict) -> dict:
    """Synchronize summary counters with the reviewed per-model capability flags.

    Args:
        versioned_payload: Capability matrix after applying bootstrap and manual rules.

    Returns:
        The same payload with refreshed summary counters.
    """
    models = versioned_payload.get("models", [])
    summary = dict(versioned_payload.get("summary", {}))
    summary["eligible_long_term"] = sum(1 for model in models if model.get("eligible_long_term"))
    summary["eligible_m4"] = sum(1 for model in models if model.get("eligible_m4"))
    versioned_payload["summary"] = summary
    return versioned_payload


def _build_versioned_capability_matrix(payload: dict) -> dict:
    """Attach bootstrap metadata so the audit snapshot becomes a versioned manifest.

    Args:
        payload: Raw capability payload generated by the static audit.

    Returns:
        A versioned capability manifest ready to be consumed by the pipeline.
    """
    versioned_payload = deepcopy(payload)
    versioned_payload["manifest_kind"] = "model_capability_matrix"
    versioned_payload["curation_status"] = "bootstrap"
    versioned_payload["review_required"] = True
    versioned_payload["eligibility_policy"] = deepcopy(ELIGIBILITY_POLICY)

    for model in versioned_payload.get("models", []):
        model.setdefault("review_status", "bootstrap")
        notes = list(model.get("notes", []))
        manifest_note = "Versioned from the static audit; review before exhaustive benchmarking."
        if manifest_note not in notes:
            notes.append(manifest_note)
        model["notes"] = notes

    versioned_payload = _apply_manual_overrides(versioned_payload)
    return _refresh_summary(versioned_payload)


def materialize_bootstrap_manifests(
    repo_root: Path,
    config_dir: Path,
    audit_output_dir: Path,
) -> dict[str, Path]:
    """Generate versioned manifests and the audit artifacts they derive from.

    Args:
        repo_root: Root directory of the benchmark repository.
        config_dir: Output directory for versioned benchmark manifests.
        audit_output_dir: Output directory for raw audit artifacts.

    Returns:
        Paths to the versioned manifests and the audit artifacts used to build them.
    """
    repo_root = repo_root.resolve()
    config_dir = config_dir.resolve()
    audit_output_dir = audit_output_dir.resolve()

    report = audit_model_library(
        repo_root=repo_root,
        submodule_root=repo_root / "s-transformers-lib",
    )
    audit_artifacts = write_audit_artifacts(report=report, output_dir=audit_output_dir)

    from .configuration import load_yaml

    # The raw audit matrix is first written to artifacts so the bootstrap step
    # has a stable inspection snapshot to version from.
    capability_payload = load_yaml(audit_artifacts["capability_matrix"])
    versioned_capability_payload = _build_versioned_capability_matrix(capability_payload)

    capability_path = write_yaml(
        versioned_capability_payload,
        config_dir / "model_capability_matrix.yaml",
    )
    datasets_path = write_yaml(
        deepcopy(DEFAULT_LONG_TERM_DATASETS),
        config_dir / "long_term_datasets.yaml",
    )
    m4_path = write_yaml(
        deepcopy(DEFAULT_M4_DATASETS),
        config_dir / "m4_datasets.yaml",
    )
    presets_path = write_yaml(
        deepcopy(DEFAULT_TRAINING_PRESETS),
        config_dir / "training_presets.yaml",
    )

    return {
        "capability_matrix": capability_path,
        "long_term_datasets": datasets_path,
        "m4_datasets": m4_path,
        "training_presets": presets_path,
        "audit_inventory": audit_artifacts["inventory"],
        "audit_contract": audit_artifacts["contract"],
        "audit_capability_matrix": audit_artifacts["capability_matrix"],
    }
