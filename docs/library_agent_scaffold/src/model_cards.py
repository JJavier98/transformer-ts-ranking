"""Model cards — Component 3 (P3) of the integration design.

A model card is the single artifact a human and an agent both read. It composes
three sources, each authored once (design doc §6):

1. **Declared** — :class:`ModelCapabilities` (Component 1) + the config schema
   (Component 2).
2. **Documented** — paper, one-line summary, selection hints (human-authored,
   loaded from ``model_docs.yaml`` when present).
3. **Empirical (runtime evidence)** — memory profile / known issues / precision,
   fed by the benchmark's ``runtime_evidence.json`` (design doc §12), carrying a
   provenance marker so declared vs measured stays distinguishable.

Target location in the library: ``s_transformers_lib/model_cards.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .interfaces.capabilities import ModelCapabilities, capabilities

# Optional human-authored docs and benchmark-fed runtime evidence. Both are
# optional so a card renders today and gains detail as these files appear.
_DOCS_FILE = Path(__file__).resolve().parent / "model_docs.yaml"
_RUNTIME_FILE = Path(__file__).resolve().parent / "runtime_evidence.json"


@dataclass
class ModelCard:
    """A complete, machine- and human-readable description of one model."""

    name: str
    family: str
    capabilities: ModelCapabilities
    summary: str = ""
    paper: str = ""
    selection_hints: dict[str, Any] = field(default_factory=dict)
    config_schema: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict (capabilities expanded) for JSON/YAML/MCP."""
        d = asdict(self)
        d["capabilities"] = asdict(self.capabilities)
        return d


def _load_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import yaml  # noqa: PLC0415

    return yaml.safe_load(path.read_text()) or {}


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def describe_model(name: str, *, with_config_schema: bool = True) -> ModelCard:
    """Assemble the :class:`ModelCard` for ``name`` from all three sources.

    Parameters
    ----------
    name : str
        Canonical model key.
    with_config_schema : bool
        If True, include the model's config JSON Schema (needs the model
        importable). Set False for a lightweight card.

    Returns
    -------
    ModelCard
    """
    caps = capabilities(name)
    docs = _load_optional_yaml(_DOCS_FILE).get(name, {})
    runtime = _load_optional_json(_RUNTIME_FILE).get(name, {})

    config_schema: dict[str, Any] = {}
    if with_config_schema:
        try:
            from .schemas import model_config_schema  # noqa: PLC0415

            config_schema = model_config_schema(name)
        except Exception:  # pragma: no cover - card still useful without it
            config_schema = {}

    return ModelCard(
        name=name,
        family=caps.family,
        capabilities=caps,
        summary=docs.get("summary", ""),
        paper=docs.get("paper", ""),
        selection_hints=docs.get("selection_hints", {}),
        config_schema=config_schema,
        runtime=runtime,  # provenance-tagged empirical evidence from the benchmark
    )
