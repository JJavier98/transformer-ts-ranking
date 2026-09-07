"""Sanity tests for the agentic scaffold logic (no library import required).

Covers the parts that do not need ``s_transformers_lib`` installed: capability
loading/filtering, JSON-Schema generation from a dataclass, model-selection
feasibility, and card assembly without the config schema. Run from the scaffold
root: ``python -m pytest tests`` (or the plain ``python tests/test_scaffold.py``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Make the scaffold importable as the package ``src`` (relative imports inside
# the modules work under any package name — that portability is the point).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interfaces.capabilities import (  # noqa: E402
    ModelCapabilities,
    capabilities,
    filter_models,
    load_capabilities,
)
from src.model_cards import describe_model  # noqa: E402
from src.schemas import dataclass_schema  # noqa: E402
from src.selection import TaskDescriptor, recommend_models  # noqa: E402


def test_all_capabilities_load():
    caps = load_capabilities()
    assert len(caps) >= 28
    assert all(isinstance(c, ModelCapabilities) for c in caps.values())
    # A known encoder-only model does not require irregular times.
    assert capabilities("patchtst").requires_irregular_times is False


def test_filter_models():
    regular = filter_models(
        lambda c: c.supports_regular_mts and not c.requires_irregular_times
    )
    assert "patchtst" in regular
    # Irregular-only models are excluded.
    irregular = filter_models(lambda c: c.requires_irregular_times)
    assert set(regular).isdisjoint(irregular)


def test_dataclass_schema():
    @dataclass
    class Cfg:
        d_model: int
        n_heads: int = 8
        name: str = "x"

    schema = dataclass_schema(Cfg)
    assert schema["type"] == "object"
    assert schema["properties"]["d_model"] == {"type": "integer"}
    assert schema["properties"]["n_heads"]["default"] == 8
    assert schema["required"] == ["d_model"]  # only the no-default field


def test_recommend_excludes_infeasible():
    task = TaskDescriptor(horizon=96, multivariate=True, irregular=False)
    recs = recommend_models(task, ranks={"patchtst": 1.0, "itransformer": 2.0}, top_k=5)
    names = [r["name"] for r in recs]
    assert names[:2] == ["patchtst", "itransformer"]  # ordered by benchmark rank
    # No irregular-only model sneaks into a regular task.
    for r in recs:
        assert capabilities(r["name"]).requires_irregular_times is False


def test_describe_model_without_library():
    card = describe_model("patchtst", with_config_schema=False)
    assert card.name == "patchtst"
    assert card.family
    assert isinstance(card.to_dict()["capabilities"], dict)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all scaffold sanity tests passed")
