"""Audit helpers for S-TransformerTS model inventory and API contracts.

The audit is intentionally static: it inspects the external model library with
AST parsing so the repository can bootstrap manifests without importing heavy
Torch/CUDA dependencies.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - fallback kept for portability
    yaml = None


REQUIRED_TRAIN_PARAMS = {"batch", "optimizer", "loss_fn"}
REQUIRED_EVAL_PARAMS = {"batch", "loss_fn"}

BOOTSTRAP_SPATIAL_MODELS = {"airformer", "earthformer", "spacetimeformer"}
BOOTSTRAP_IRREGULAR_MODELS = {"contiformer", "tpatchgnn"}
BOOTSTRAP_EXOGENOUS_MODELS = {"chronos2", "tft", "timexer"}

TIME_MARK_HINTS = {"x_mark", "x_mark_enc", "x_mark_dec", "y_mark", "y_mark_dec"}
SEQ2SEQ_HINTS = {"x_dec", "x_mark", "x_mark_enc", "x_mark_dec", "y_mark"}
IRREGULAR_HINTS = {"x_time", "x_mask", "pred_time"}

BOOTSTRAP_NOTE = (
    "Bootstrap capability matrix uses deterministic heuristics and requires manual review "
    "before the full benchmark."
)


@dataclass
class ModelAuditEntry:
    """Static audit evidence collected for one model.

    Attributes summarize registry presence, source inspection results, inferred
    adapter family and bootstrap eligibility flags.
    """

    model_name: str
    in_filesystem: bool
    registered: bool
    model_module_path: str
    model_class_name: str | None = None
    config_class_name: str | None = None
    source_inspected: bool = False
    inspection_error: str | None = None
    config_declared: bool | None = None
    config_instantiable: bool | None = None
    config_error: str | None = None
    subclasses_base_model: bool | None = None
    forward_params: list[str] = field(default_factory=list)
    train_step_params: list[str] = field(default_factory=list)
    eval_step_params: list[str] = field(default_factory=list)
    missing_train_step_params: list[str] = field(default_factory=list)
    missing_eval_step_params: list[str] = field(default_factory=list)
    supports_regular_mts: bool | None = None
    supports_univariate: bool | None = None
    requires_time_marks: bool | None = None
    requires_exogenous: bool | None = None
    requires_irregular_times: bool | None = None
    requires_spatial_structure: bool | None = None
    eligible_long_term: bool | None = None
    eligible_m4: bool | None = None
    adapter_name: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Full model-library audit report.

    Attributes:
        generated_at: UTC timestamp of the audit.
        repo_root: Root of the benchmark repository.
        submodule_root: Root of the S-TransformerTS submodule.
        filesystem_only: Models found on disk but not in the public registry.
        registry_only: Models declared in the registry but missing on disk.
        summary: Aggregate counters for the audit.
        models: Per-model audit entries.
    """

    generated_at: str
    repo_root: str
    submodule_root: str
    filesystem_only: list[str]
    registry_only: list[str]
    summary: dict[str, int]
    models: list[ModelAuditEntry]


def _discover_model_directories(models_dir: Path) -> list[str]:
    """List model package directories present on disk.

    Args:
        models_dir: ``src/models`` directory inside the submodule.

    Returns:
        Sorted model directory names that look like Python packages.
    """
    return sorted(
        entry.name
        for entry in models_dir.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists()
    )


def _parse_python_module(module_path: Path) -> ast.Module:
    """Parse a Python source file into an AST module.

    Args:
        module_path: Python file to parse.

    Returns:
        The parsed AST module.
    """
    return ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))


def _extract_registry_mapping(module_ast: ast.Module, mapping_name: str) -> dict[str, str]:
    """Extract a registry dict literal from the registry module AST.

    Args:
        module_ast: Parsed AST of ``registry.py``.
        mapping_name: Name of the dict literal to extract.

    Returns:
        A string-to-string mapping from registry key to exported class name.
    """
    for node in module_ast.body:
        value: ast.AST | None = None

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == mapping_name:
                    value = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == mapping_name:
                value = node.value

        if value is None:
            continue

        parsed = ast.literal_eval(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{mapping_name} is not a dict literal in registry.py")
        return {str(key): str(item) for key, item in parsed.items()}

    raise ValueError(f"{mapping_name} was not found in registry.py")


def _load_registry_mappings(registry_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load model and config registries from the submodule registry file.

    Args:
        registry_path: Path to ``src/models/registry.py``.

    Returns:
        The model registry and config registry mappings.
    """
    registry_ast = _parse_python_module(registry_path)
    return (
        _extract_registry_mapping(registry_ast, "MODEL_REGISTRY"),
        _extract_registry_mapping(registry_ast, "CONFIG_REGISTRY"),
    )


def _find_class_def(module_ast: ast.Module, class_name: str) -> ast.ClassDef | None:
    """Find one class definition by name inside a parsed module.

    Args:
        module_ast: Parsed AST module to inspect.
        class_name: Class name to locate.

    Returns:
        The matching class definition, or ``None`` when it is absent.
    """
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _resolve_model_alias(model_dir: Path, exported_name: str) -> str | None:
    """Resolve re-exported model names declared in ``__init__.py``.

    Args:
        model_dir: Model package directory.
        exported_name: Public class name used by the registry.

    Returns:
        The concrete class name defined in ``model.py``, if it can be resolved.
    """
    init_path = model_dir / "__init__.py"
    if not init_path.exists():
        return None

    try:
        init_ast = _parse_python_module(init_path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    for node in init_ast.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 1 or node.module != "model":
            continue
        for alias in node.names:
            if alias.asname == exported_name:
                return alias.name

    return None


def _base_name(expr: ast.expr) -> str | None:
    """Recover the dotted base name from an AST expression.

    Args:
        expr: AST expression describing a class base.

    Returns:
        The normalized dotted base name, or ``None`` if it cannot be recovered.
    """
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        prefix = _base_name(expr.value)
        return f"{prefix}.{expr.attr}" if prefix else expr.attr
    if isinstance(expr, ast.Subscript):
        return _base_name(expr.value)
    return None


def _extract_argument_names(arguments: ast.arguments) -> list[str]:
    """Extract every declared argument name except ``self``.

    Args:
        arguments: AST arguments object from a function definition.

    Returns:
        Argument names in declaration order.
    """
    names: list[str] = []
    for arg in list(arguments.posonlyargs) + list(arguments.args):
        if arg.arg != "self":
            names.append(arg.arg)
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    for arg in arguments.kwonlyargs:
        if arg.arg != "self":
            names.append(arg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return names


def _required_argument_names(arguments: ast.arguments) -> list[str]:
    """Extract required argument names from an ``__init__`` signature.

    Args:
        arguments: AST arguments object from a function definition.

    Returns:
        Required positional and keyword-only argument names, excluding ``self``.
    """
    positional = [
        arg.arg
        for arg in list(arguments.posonlyargs) + list(arguments.args)
        if arg.arg != "self"
    ]
    positional_defaults = len(arguments.defaults)
    required_positional = positional[: len(positional) - positional_defaults]
    required_kwonly = [
        arg.arg
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if default is None and arg.arg != "self"
    ]
    return required_positional + required_kwonly


def _append_note(entry: ModelAuditEntry, note: str) -> None:
    """Append a note only once to an audit entry.

    Args:
        entry: Audit entry to mutate.
        note: Note to append when it is not already present.
    """
    if note not in entry.notes:
        entry.notes.append(note)


def _classify_adapter(entry: ModelAuditEntry) -> str:
    """Infer the adapter family suggested by the static audit evidence.

    Args:
        entry: Model audit entry with collected signature hints.

    Returns:
        The inferred adapter family name.
    """
    params = set(entry.forward_params) | set(entry.train_step_params) | set(entry.eval_step_params)

    if entry.requires_spatial_structure:
        return "spatial"
    if entry.requires_irregular_times:
        return "irregular"
    if entry.requires_exogenous:
        return "exogenous_aware"
    if params & SEQ2SEQ_HINTS or entry.requires_time_marks:
        return "seq2seq"
    return "encoder_only"


def _inspect_model_source(model_dir: Path, entry: ModelAuditEntry) -> None:
    """Inspect one model package statically and populate the audit entry.

    Args:
        model_dir: Model package directory to inspect.
        entry: Audit entry to populate in place.
    """
    model_path = model_dir / "model.py"
    config_path = model_dir / "config.py"

    if entry.model_class_name is None:
        entry.inspection_error = "Missing model class name in registry."
    elif not model_path.exists():
        entry.inspection_error = f"Missing file: {model_path.name}"
    else:
        try:
            model_ast = _parse_python_module(model_path)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            entry.inspection_error = f"{type(exc).__name__}: {exc}"
        else:
            class_def = _find_class_def(model_ast, entry.model_class_name)
            if class_def is None:
                aliased_name = _resolve_model_alias(model_dir, entry.model_class_name)
                if aliased_name is not None:
                    class_def = _find_class_def(model_ast, aliased_name)
            if class_def is None:
                entry.inspection_error = (
                    f"Class {entry.model_class_name} was not found in {model_path.name}"
                )
            else:
                entry.source_inspected = True
                base_names = {
                    name
                    for name in (_base_name(base) for base in class_def.bases)
                    if name is not None
                }
                entry.subclasses_base_model = any(
                    name.endswith("BaseTransformerModel") for name in base_names
                )

                methods = {
                    node.name: node
                    for node in class_def.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if "forward" in methods:
                    entry.forward_params = _extract_argument_names(methods["forward"].args)
                if "train_step" in methods:
                    entry.train_step_params = _extract_argument_names(methods["train_step"].args)
                if "eval_step" in methods:
                    entry.eval_step_params = _extract_argument_names(methods["eval_step"].args)

                entry.missing_train_step_params = sorted(
                    REQUIRED_TRAIN_PARAMS.difference(entry.train_step_params)
                )
                entry.missing_eval_step_params = sorted(
                    REQUIRED_EVAL_PARAMS.difference(entry.eval_step_params)
                )

    if entry.config_class_name is None:
        entry.config_declared = False
        entry.config_error = "Missing config class name in registry."
        return

    if not config_path.exists():
        entry.config_declared = False
        entry.config_error = f"Missing file: {config_path.name}"
        return

    try:
        config_ast = _parse_python_module(config_path)
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        entry.config_declared = False
        entry.config_error = f"{type(exc).__name__}: {exc}"
        return

    config_class = _find_class_def(config_ast, entry.config_class_name)
    if config_class is None:
        entry.config_declared = False
        entry.config_error = f"Class {entry.config_class_name} was not found in {config_path.name}"
        return

    entry.config_declared = True
    init_method = next(
        (
            node
            for node in config_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
        ),
        None,
    )

    if init_method is None:
        entry.config_instantiable = None
        entry.config_error = "Static audit did not verify runtime config instantiation."
        return

    required_args = _required_argument_names(init_method.args)
    entry.config_instantiable = len(required_args) == 0
    if required_args:
        entry.config_error = f"Config __init__ requires: {', '.join(required_args)}"


def _apply_bootstrap_capabilities(entry: ModelAuditEntry) -> None:
    """Infer bootstrap capability flags from static audit evidence.

    Args:
        entry: Audit entry to enrich in place.
    """
    params = set(entry.forward_params) | set(entry.train_step_params) | set(entry.eval_step_params)

    entry.requires_irregular_times = (
        entry.model_name in BOOTSTRAP_IRREGULAR_MODELS or bool(params & IRREGULAR_HINTS)
    )
    entry.requires_spatial_structure = entry.model_name in BOOTSTRAP_SPATIAL_MODELS
    entry.requires_exogenous = entry.model_name in BOOTSTRAP_EXOGENOUS_MODELS
    entry.requires_time_marks = bool(params & TIME_MARK_HINTS) and not entry.requires_irregular_times
    entry.supports_regular_mts = bool(
        entry.source_inspected and not entry.requires_irregular_times and not entry.requires_spatial_structure
    )
    entry.supports_univariate = None
    entry.adapter_name = (
        _classify_adapter(entry)
        if entry.registered and entry.source_inspected
        else "manual_review"
    )
    entry.eligible_long_term = bool(
        entry.registered
        and entry.source_inspected
        and entry.supports_regular_mts
        and not entry.requires_exogenous
    )
    entry.eligible_m4 = bool(
        entry.registered
        and entry.source_inspected
        and not entry.requires_irregular_times
        and not entry.requires_spatial_structure
        and not entry.requires_exogenous
    )

    # Notes preserve the reasoning trail that later manual review uses to accept
    # or override the bootstrap decision.
    _append_note(entry, BOOTSTRAP_NOTE)
    if not entry.in_filesystem:
        _append_note(entry, "Registered model is missing a filesystem directory.")
    if not entry.registered:
        _append_note(entry, "Filesystem model is missing a registry entry.")
    if entry.inspection_error:
        _append_note(entry, f"Source inspection failed: {entry.inspection_error}")
    if entry.missing_train_step_params:
        _append_note(
            entry,
            f"train_step is missing required params: {', '.join(entry.missing_train_step_params)}.",
        )
    if entry.missing_eval_step_params:
        _append_note(
            entry,
            f"eval_step is missing required params: {', '.join(entry.missing_eval_step_params)}.",
        )
    if entry.config_declared is False and entry.config_error:
        _append_note(entry, f"Config declaration issue: {entry.config_error}")
    if entry.config_instantiable is False and entry.config_error:
        _append_note(entry, f"Config instantiation issue: {entry.config_error}")
    if entry.config_instantiable is None and entry.config_declared:
        _append_note(entry, "Config runtime instantiation was not attempted in the static audit.")


def _collect_contract_issues(entry: ModelAuditEntry) -> list[str]:
    """Translate raw audit evidence into normalized contract issue codes.

    Args:
        entry: Model audit entry to inspect.

    Returns:
        Stable issue codes used by the contract report.
    """
    issues: list[str] = []

    if not entry.in_filesystem:
        issues.append("missing_filesystem_directory")
    if not entry.registered:
        issues.append("missing_registry_entry")
    if entry.registered and not entry.source_inspected:
        issues.append("model_source_inspection_failed")
    if entry.subclasses_base_model is False:
        issues.append("not_subclass_of_base_transformer_model")
    if entry.missing_train_step_params:
        issues.append("train_step_contract_mismatch")
    if entry.missing_eval_step_params:
        issues.append("eval_step_contract_mismatch")
    if entry.config_declared is False:
        issues.append("config_declaration_missing")
    if entry.config_instantiable is False:
        issues.append("config_default_instantiation_failed")

    return issues


def audit_model_library(repo_root: Path, submodule_root: Path) -> AuditReport:
    """Audit the external model library statically.

    Args:
        repo_root: Root directory of the benchmark repository.
        submodule_root: Root directory of the S-TransformerTS submodule.

    Returns:
        The full static audit report.
    """
    repo_root = repo_root.resolve()
    submodule_root = submodule_root.resolve()
    models_dir = submodule_root / "src" / "models"
    registry_path = models_dir / "registry.py"

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    filesystem_models = _discover_model_directories(models_dir)
    model_registry, config_registry = _load_registry_mappings(registry_path)
    registered_models = sorted(model_registry)
    all_models = sorted(set(filesystem_models) | set(registered_models))
    entries: list[ModelAuditEntry] = []

    for model_name in all_models:
        entry = ModelAuditEntry(
            model_name=model_name,
            in_filesystem=model_name in filesystem_models,
            registered=model_name in registered_models,
            model_module_path=f"src/models/{model_name}",
            model_class_name=model_registry.get(model_name),
            config_class_name=config_registry.get(model_name),
        )
        _inspect_model_source(models_dir / model_name, entry)
        _apply_bootstrap_capabilities(entry)
        entries.append(entry)

    filesystem_only = sorted(set(filesystem_models) - set(registered_models))
    registry_only = sorted(set(registered_models) - set(filesystem_models))
    summary = {
        "total_models": len(entries),
        "filesystem_models": len(filesystem_models),
        "registered_models": len(registered_models),
        "filesystem_only": len(filesystem_only),
        "registry_only": len(registry_only),
        "inspection_failures": sum(
            1 for entry in entries if entry.registered and not entry.source_inspected
        ),
        "eligible_long_term": sum(1 for entry in entries if entry.eligible_long_term),
        "eligible_m4": sum(1 for entry in entries if entry.eligible_m4),
    }

    return AuditReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        repo_root=str(repo_root),
        submodule_root=str(submodule_root),
        filesystem_only=filesystem_only,
        registry_only=registry_only,
        summary=summary,
        models=entries,
    )


def write_audit_artifacts(report: AuditReport, output_dir: Path) -> dict[str, Path]:
    """Write the audit report into machine-readable benchmark artifacts.

    Args:
        report: Full static audit report.
        output_dir: Directory where audit artifacts should be written.

    Returns:
        Paths to the generated audit artifacts.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / "model_inventory.json"
    contract_path = output_dir / "api_contract_report.json"
    matrix_path = output_dir / "model_capability_matrix.yaml"

    inventory_payload = {
        "generated_at": report.generated_at,
        "repo_root": report.repo_root,
        "submodule_root": report.submodule_root,
        "summary": report.summary,
        "filesystem_only": report.filesystem_only,
        "registry_only": report.registry_only,
        "models": [asdict(model) for model in report.models],
    }

    contract_payload = {
        "generated_at": report.generated_at,
        "summary": report.summary,
        "models": [
            {
                "model_name": model.model_name,
                "registered": model.registered,
                "in_filesystem": model.in_filesystem,
                "source_inspected": model.source_inspected,
                "inspection_error": model.inspection_error,
                "subclasses_base_model": model.subclasses_base_model,
                "forward_params": model.forward_params,
                "train_step_params": model.train_step_params,
                "eval_step_params": model.eval_step_params,
                "config_declared": model.config_declared,
                "config_instantiable": model.config_instantiable,
                "issues": _collect_contract_issues(model),
                "notes": model.notes,
            }
            for model in report.models
        ],
    }

    capability_payload = {
        "generated_at": report.generated_at,
        "summary": report.summary,
        "models": [
            {
                "model_name": model.model_name,
                "registered": model.registered,
                "supports_regular_mts": model.supports_regular_mts,
                "supports_univariate": model.supports_univariate,
                "requires_time_marks": model.requires_time_marks,
                "requires_exogenous": model.requires_exogenous,
                "requires_irregular_times": model.requires_irregular_times,
                "requires_spatial_structure": model.requires_spatial_structure,
                "eligible_long_term": model.eligible_long_term,
                "eligible_m4": model.eligible_m4,
                "adapter_name": model.adapter_name,
                "notes": model.notes,
            }
            for model in report.models
        ],
    }

    inventory_path.write_text(json.dumps(inventory_payload, indent=2), encoding="utf-8")
    contract_path.write_text(json.dumps(contract_payload, indent=2), encoding="utf-8")

    if yaml is not None:
        matrix_path.write_text(
            yaml.safe_dump(capability_payload, sort_keys=False),
            encoding="utf-8",
        )
    else:  # pragma: no cover - fallback kept for portability
        matrix_path.write_text(json.dumps(capability_payload, indent=2), encoding="utf-8")

    return {
        "inventory": inventory_path,
        "contract": contract_path,
        "capability_matrix": matrix_path,
    }