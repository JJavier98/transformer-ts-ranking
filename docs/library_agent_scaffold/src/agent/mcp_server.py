"""MCP server — Component 4 (P5) of the integration design.

Exposes the library to agents over the Model Context Protocol as an **optional
extra** (``s-transformers-lib[agent]``). Tools reuse ``create_model`` and the
existing ``fit``/``predict`` — no reimplementation. Long-running training is
deliberately NOT a blocking tool (design doc §15); the default surface is
inference + discovery.

Run: ``python -m s_transformers_lib.agent.mcp_server``
(installed via ``pip install s-transformers-lib[agent]``).

Target location in the library: ``s_transformers_lib/agent/mcp_server.py``.

Resources
---------
- ``stlib://models``           list of models + summaries
- ``stlib://models/{name}``    the full model card
- ``stlib://capabilities``     the capability matrix (intrinsic fields)

Tools
-----
- ``list_models``       optionally filtered by a capability predicate name
- ``describe_model``    the model card for one model
- ``recommend_models``  ranked candidates for a task descriptor
- ``forecast``          run inference with a (schema-validated) ForecastInput
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional extra
    raise SystemExit(
        "The MCP server needs the 'agent' extra: pip install s-transformers-lib[agent]"
    ) from exc

from ..interfaces.capabilities import load_capabilities
from ..model_cards import describe_model
from ..selection import TaskDescriptor, recommend_models

mcp = FastMCP("s-transformers-lib")


# --- Resources -------------------------------------------------------------
@mcp.resource("stlib://models")
def models_resource() -> str:
    """List every model with its family and one-line summary."""
    out = {}
    for name in sorted(load_capabilities()):
        card = describe_model(name, with_config_schema=False)
        out[name] = {"family": card.family, "summary": card.summary}
    return json.dumps(out, indent=2)


@mcp.resource("stlib://models/{name}")
def model_card_resource(name: str) -> str:
    """The full model card for ``name`` (capabilities + docs + runtime)."""
    return json.dumps(describe_model(name).to_dict(), indent=2, default=str)


@mcp.resource("stlib://capabilities")
def capabilities_resource() -> str:
    """The intrinsic capability matrix for all models."""
    return json.dumps(
        {n: asdict(c) for n, c in load_capabilities().items()}, indent=2
    )


# --- Tools -----------------------------------------------------------------
@mcp.tool()
def list_models() -> list[str]:
    """Return the names of all registered models."""
    return sorted(load_capabilities())


@mcp.tool()
def describe(name: str) -> dict[str, Any]:
    """Return the model card for ``name``."""
    return describe_model(name).to_dict()


@mcp.tool()
def recommend(
    horizon: int,
    multivariate: bool = True,
    exogenous: bool = False,
    irregular: bool = False,
    univariate: bool = False,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Recommend models for a task described by its shape/requirements."""
    task = TaskDescriptor(
        horizon=horizon,
        multivariate=multivariate,
        exogenous=exogenous,
        irregular=irregular,
        univariate=univariate,
    )
    return recommend_models(task, top_k=top_k)


@mcp.tool()
def forecast(name: str, forecast_input: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run inference with ``name`` on a ForecastInput payload.

    ``forecast_input`` follows ``schemas.forecast_input_schema()``; tensor
    fields are bound by the host. Returns the prediction shape and values.
    """
    import torch  # noqa: PLC0415

    from s_transformers_lib.interfaces.forecasting import ForecastInput  # noqa: PLC0415
    from s_transformers_lib.models import create_model  # noqa: PLC0415

    model = create_model(name, config=config or {})
    model.eval()
    tensors = {
        k: torch.as_tensor(v["values"]) if isinstance(v, dict) and "values" in v else v
        for k, v in forecast_input.items()
    }
    with torch.no_grad():
        out = model.predict(ForecastInput(**tensors))
    pred = out.prediction
    return {"shape": list(pred.shape), "prediction": pred.tolist()}


def main() -> None:
    """Entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
