# Agentic integration scaffold for `s-transformers-lib`

A **portable, working implementation** of the agentic layer from
[`docs/agentic_integration_design.md`](../agentic_integration_design.md), built here in the
benchmark repo to carry into the **library**. It touches no benchmark runtime code, no `src/`, no
SLURM scripts, no `.venv`, and **not the `s-transformers-lib` submodule** — so it cannot affect
experiments. (The library is off-limits from this repo; this mirrors the docs-scaffold precedent.)

## What is implemented (design-doc components)

| File (`src/…`) | Component | Status |
|---|---|---|
| `interfaces/capabilities.py` + `capabilities.yaml` | **1 (P1)** capabilities | ✅ `ModelCapabilities`, `capabilities()`, `filter_models()`; values extracted from the benchmark's validated matrix (31 models) |
| `schemas.py` | **2 (P2)** JSON Schema | ✅ dataclass→schema; `forecast_input_schema` / `training_config_schema` / `model_config_schema` |
| `model_cards.py` + `model_docs.yaml` | **3 (P3)** model cards | ✅ `describe_model()` composing declared + documented + runtime evidence |
| `agent/mcp_server.py` | **4 (P5)** MCP server | ✅ resources (`stlib://models`, `/{name}`, `/capabilities`) + tools (`list_models`, `describe`, `recommend`, `forecast`) |
| `selection.py` | **5 (P6)** selection | ✅ `recommend_models()` (capability feasibility + benchmark ranks) |

**Not included** (separate, non-agent components of the same doc): P7 model-aware dataloaders and P8
profiling primitives — see design doc §9/§10. **P4** (`runtime_evidence.json` from the benchmark)
is the seam that fills the cards' `runtime:` block; export it from this repo when v1 is frozen.

Sanity-tested (no library needed): `python tests/test_scaffold.py` → 5/5 pass.

## How to adopt it in the library repo

1. Copy `src/*` into the library package (target: `s_transformers_lib/`), so:
   `interfaces/capabilities.py`, `schemas.py`, `model_cards.py`, `selection.py`,
   `agent/mcp_server.py`, plus `capabilities.yaml` / `model_docs.yaml`.
2. Merge `pyproject.agent-extra.toml` (the `[agent]` extra + the console entry point).
3. Re-export the public API from the package: `capabilities`, `describe_model`, `recommend_models`.
4. Verify: `pip install ".[agent]"` then `python -m s_transformers_lib.agent.mcp_server`.

## Important on adoption

- **Distribute capabilities into the models (end state).** `capabilities.yaml` is the *interim*
  source. The design's end state (§4) is each model's `config.py` declaring its own
  `ModelCapabilities`, aggregated by the registry; then `capabilities.yaml` becomes a fallback.
  Until distributed, `is_pretrained_zeroshot` and `family` should be verified per model.
- **⚠️ Benchmark parity gate (design doc §12/§14).** Adopting P1 changes where the benchmark could
  source capabilities. Do **not** let this change the benchmark's eligible model set silently: if the
  benchmark later consumes `capabilities()`, assert the eligible set is byte-identical first, and
  never during a run. This scaffold does not touch the benchmark, so building it now is safe; the
  parity gate applies only when the benchmark switches to consume it.
- **Package import path.** Modules import `s_transformers_lib.*`; adjust if the library's packaging
  differs.
- **`train` is intentionally not an MCP tool** (long/costly, §15). The default agent surface is
  discovery + inference (`forecast`).
