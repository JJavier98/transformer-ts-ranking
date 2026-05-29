"""Command-line interface for ranking utilities.

The CLI exposes the current pipeline stages individually so the repository can
be validated incrementally: audit, manifest materialization, smoke checks and
runtime compatibility review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import materialize_bootstrap_manifests
from .discovery.model_audit import audit_model_library, write_audit_artifacts
from .discovery.runtime_compatibility import probe_model_compatibility
from .evaluation.smoke import run_long_term_smoke, run_m4_smoke


def _default_repo_root() -> Path:
    """Return the repository root inferred from the installed package path.

    Returns:
        The root directory of the current workspace checkout.
    """
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser and all supported subcommands.

    Returns:
        The fully configured argument parser for the benchmark utilities.
    """
    parser = argparse.ArgumentParser(
        prog="transformer-ts-ranking",
        description="Benchmarking and audit utilities for S-TransformerTS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit-models",
        help="Audit the S-TransformerTS model inventory and API contract.",
    )
    audit_parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root for transformer-ts-ranking.",
    )
    audit_parser.add_argument(
        "--submodule-dir",
        type=Path,
        default=None,
        help="Path to the S-TransformerTS git submodule.",
    )
    audit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated audit artifacts.",
    )

    manifests_parser = subparsers.add_parser(
        "materialize-manifests",
        help="Write versioned benchmark manifests from the static audit.",
    )
    manifests_parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root for transformer-ts-ranking.",
    )
    manifests_parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Target directory for versioned benchmark manifests.",
    )
    manifests_parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=None,
        help="Directory for generated audit artifacts used as manifest source.",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-long-term",
        help="Run a data-centric smoke plan for the long-term benchmark.",
    )
    smoke_parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root for transformer-ts-ranking.",
    )
    smoke_parser.add_argument(
        "--dataset",
        default="ETTh1",
        help="Long-term dataset name to inspect in the smoke plan.",
    )
    smoke_parser.add_argument(
        "--preset",
        default="smoke",
        help="Training preset to use for the smoke plan.",
    )
    smoke_parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing benchmark manifests.",
    )
    smoke_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated smoke-plan artifact.",
    )

    m4_smoke_parser = subparsers.add_parser(
        "smoke-m4",
        help="Run a data-centric smoke plan for one M4 frequency slice.",
    )
    m4_smoke_parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root for transformer-ts-ranking.",
    )
    m4_smoke_parser.add_argument(
        "--frequency",
        default="Hourly",
        help="M4 frequency label to inspect in the smoke plan.",
    )
    m4_smoke_parser.add_argument(
        "--preset",
        default="smoke",
        help="Training preset to use for the smoke plan.",
    )
    m4_smoke_parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing benchmark manifests.",
    )
    m4_smoke_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated smoke-plan artifact.",
    )

    probe_parser = subparsers.add_parser(
        "probe-compatibility",
        help="Probe runtime compatibility of models for long-term and M4 tasks.",
    )
    probe_parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root for transformer-ts-ranking.",
    )
    probe_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names. Defaults to all registered models.",
    )
    probe_parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Target JSON path for the compatibility report.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the requested CLI command.

    Args:
        argv: Optional command-line arguments. When ``None``, argparse reads
            directly from ``sys.argv``.

    Returns:
        Standard process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()

    if args.command == "audit-models":
        submodule_dir = (
            args.submodule_dir.resolve()
            if args.submodule_dir is not None
            else repo_root / "s-transformers-lib"
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else repo_root / "artifacts" / "audit"
        )

        report = audit_model_library(repo_root=repo_root, submodule_root=submodule_dir)
        artifact_paths = write_audit_artifacts(report=report, output_dir=output_dir)

        print(f"Audited {report.summary['total_models']} models")
        print(f"Registered models: {report.summary['registered_models']}")
        print(f"Filesystem models: {report.summary['filesystem_models']}")
        print(f"Inspection failures: {report.summary['inspection_failures']}")
        print(f"Filesystem only: {report.summary['filesystem_only']}")
        print(f"Registry only: {report.summary['registry_only']}")
        print(f"Eligible long-term: {report.summary['eligible_long_term']}")
        print(f"Eligible M4: {report.summary['eligible_m4']}")
        print("Artifacts:")
        for label, path in artifact_paths.items():
            print(f"  - {label}: {path}")
        return 0

    if args.command == "materialize-manifests":
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else repo_root / "configs" / "benchmark"
        )
        audit_output_dir = (
            args.audit_output_dir.resolve()
            if args.audit_output_dir is not None
            else repo_root / "artifacts" / "audit"
        )
        manifest_paths = materialize_bootstrap_manifests(
            repo_root=repo_root,
            config_dir=config_dir,
            audit_output_dir=audit_output_dir,
        )
        print("Manifests:")
        for label, path in manifest_paths.items():
            print(f"  - {label}: {path}")
        return 0

    if args.command == "smoke-long-term":
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else repo_root / "configs" / "benchmark"
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else repo_root / "artifacts" / "smoke"
        )
        payload, artifact_path = run_long_term_smoke(
            repo_root=repo_root,
            dataset_name=args.dataset,
            preset_name=args.preset,
            config_dir=config_dir,
            output_dir=output_dir,
        )
        runtime_status = payload["runtime"]["available"]
        print(f"Smoke dataset: {payload['dataset']['dataset_name']}")
        print(f"Selected models: {', '.join(payload['selected_models'])}")
        print(f"Runtime available: {runtime_status}")
        print(f"Artifact: {artifact_path}")
        return 0

    if args.command == "smoke-m4":
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else repo_root / "configs" / "benchmark"
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else repo_root / "artifacts" / "smoke"
        )
        payload, artifact_path = run_m4_smoke(
            repo_root=repo_root,
            frequency_label=args.frequency,
            preset_name=args.preset,
            config_dir=config_dir,
            output_dir=output_dir,
        )
        runtime_status = payload["runtime"]["available"]
        print(f"Smoke frequency: {payload['dataset']['frequency_label']}")
        print(f"Selected models: {', '.join(payload['selected_models'])}")
        print(f"Naive2 reference OWA: {payload['naive2_reference']['mean_owa']:.3f}")
        print(f"Runtime available: {runtime_status}")
        print(f"Artifact: {artifact_path}")
        return 0

    if args.command == "probe-compatibility":
        output_path = (
            args.output_path.resolve()
            if args.output_path is not None
            else repo_root / "artifacts" / "review" / "runtime_compatibility.json"
        )
        models = None
        if args.models:
            # The CLI accepts a comma-separated list to keep ad hoc subset probes cheap.
            models = [model_name.strip() for model_name in args.models.split(",") if model_name.strip()]
        payload, artifact_path = probe_model_compatibility(
            repo_root=repo_root,
            output_path=output_path,
            models=models,
        )
        print(f"Probed models: {payload['summary']['total_models']}")
        print(f"Long-term compatible: {payload['summary']['long_term_compatible']}")
        print(f"M4 compatible: {payload['summary']['m4_compatible']}")
        if "long_term_fit_ok" in payload["summary"]:
            print(f"Long-term fit() ok: {payload['summary']['long_term_fit_ok']}")
        if "m4_fit_ok" in payload["summary"]:
            print(f"M4 fit() ok: {payload['summary']['m4_fit_ok']}")
        print(f"Both compatible: {payload['summary']['both_compatible']}")
        if "both_fit_ok" in payload["summary"]:
            print(f"Both fit() ok: {payload['summary']['both_fit_ok']}")
        print(f"Artifact: {artifact_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
