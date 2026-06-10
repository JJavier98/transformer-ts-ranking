---
name: run-transformer-ts-ranking
description: Run, test, audit, and drive transformer-ts-ranking. Use when asked to start the benchmark pipeline, run CLI commands, audit models, smoke-test, probe compatibility, validate forward passes, or run the test suite.
---

`transformer-ts-ranking` is a Python CLI / library for benchmarking S-TransformerTS models on long-term and M4 forecasting datasets. It has no GUI. The agent path is `.claude/skills/run-transformer-ts-ranking/smoke.sh`, which drives all six CLI subcommands via `runner.py`. All paths below are relative to the repo root.

## Prerequisites

No extra system packages needed. The conda environment `torch_env` must exist (it's pre-populated in this repo).

```bash
conda env list   # confirm torch_env is listed
```

## Setup

The package is **not installed** into `torch_env` — it must be added to `sys.path` at runtime. `runner.py` handles this automatically; never call `python -m transformer_ts_ranking` directly without setting PYTHONPATH first.

## Run (agent path)

The smoke driver exercises every CLI subcommand in sequence:

```bash
bash .claude/skills/run-transformer-ts-ranking/smoke.sh
```

Expected output ends with `All smoke checks passed.` Artifacts land under `artifacts/` and `configs/benchmark/`.

To call a single subcommand:

```bash
conda run -n torch_env python .claude/skills/run-transformer-ts-ranking/runner.py <subcommand> [args]
```

Available subcommands:

| subcommand | what it does | key flags |
|---|---|---|
| `audit-models` | Inventory S-TransformerTS models vs. registry, write `artifacts/audit/` | — |
| `materialize-manifests` | Stamp versioned YAML manifests into `configs/benchmark/` | — |
| `smoke-long-term` | Data-centric smoke plan for a long-term dataset | `--dataset ETTh1` |
| `smoke-m4` | Data-centric smoke plan for an M4 frequency slice | `--frequency Hourly` |
| `probe-compatibility` | Runtime-probe fit()/predict() for models | `--models patchtst,autoformer` |
| `validate-canonical-forward` | Validate canonical predict() shapes | `--models patchtst,autoformer` |

Pass `--models` as a comma-separated subset to keep probe/validate fast (~5 s vs. minutes for all 29).

## Test

```bash
conda run -n torch_env python -c "
import sys, pytest
sys.path.insert(0, 'src')
sys.exit(pytest.main(['-q', '--tb=short', 'tests/']))
"
```

16 tests, ~2 minutes. All pass. The FutureWarning about `pynvml` on stderr is harmless.

## Gotchas

- **`conda run` does not forward stdin heredocs.** `conda run -n torch_env python -` with `<<'EOF'` silently swallows stdin and produces no output. Always pass a file path (`runner.py`) instead of a heredoc script.
- **Package not installed.** `python -m transformer_ts_ranking` fails with `No module named transformer_ts_ranking` unless `src/` is on `sys.path`. Use `runner.py` or pass `-c "import sys; sys.path.insert(0, 'src'); ..."`.
- **`probe-compatibility` / `validate-canonical-forward` without `--models` probe all 29 models** and take several minutes. Always pass `--models patchtst,autoformer` (or another subset) for quick checks.
- **pynvml FutureWarning** appears on every `conda run` invocation — it comes from torch's CUDA init, is harmless, and can be suppressed with `2>/dev/null` or `-W ignore::FutureWarning`.

## Troubleshooting

- **`No module named transformer_ts_ranking`**: You called `python -m transformer_ts_ranking` without `src/` on the path. Use `runner.py` instead.
- **`conda run ... python -` produces no output / exit 0 silently**: heredoc stdin not forwarded through conda run — write a script file and pass its path.
- **Tests fail to collect with `ModuleNotFoundError`**: `pytest tests/` without the `sys.path.insert(0, 'src')` wrapper fails. Use the `python -c "import sys, pytest; sys.path.insert(0, 'src'); ..."` form shown above.
