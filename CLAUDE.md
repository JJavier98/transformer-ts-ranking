# transformer-ts-ranking — Project Context

## What this project is

A reproducible benchmark pipeline that evaluates and ranks every transformer model in `s-transformers-lib/` on time-series forecasting tasks. The library (`s-transformers-lib/`) is developed by the same author and exposes 29 state-of-the-art transformer models for time series forecasting through a unified sklearn-style API (`config → model → fit() → predict()`). This repo consumes that library as a git submodule and **must never modify its source**. The library is publicly available at https://github.com/ari-dasci/S-TransformerTS.

### Dual purpose

1. **Benchmark & ranking** — produce reproducible performance tables (accuracy + efficiency) across all eligible models.
2. **Scientific paper material** — the benchmark is the empirical foundation for a tutorial/survey paper that *markets* `s-transformers-lib` as a unified, practical toolkit for the time series forecasting community. Every figure, table, and statistical test must be publication-ready.

### Benchmark tracks

- **Long-term forecast** — 9 datasets × 4 horizons (ETTh1/h2, ETTm1/m2, weather, electricity, traffic, exchange_rate, illness)
- **Short-term forecast (M4)** — 6 frequency slices (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly), ranked by OWA

## Environment

All benchmark commands, tests, and CLI invocations **must use the `torch_env` conda environment**:

```bash
conda run -n torch_env python .claude/skills/run-transformer-ts-ranking/runner.py <subcommand> [flags]
```

The package is **not installed** into `torch_env`. The `runner.py` shim adds `src/` to `sys.path`. Never call `python -m transformer_ts_ranking` directly; it will fail with `ModuleNotFoundError`.

Tests (16, ~2 min):
```bash
conda run -n torch_env python -c "import sys, pytest; sys.path.insert(0, 'src'); sys.exit(pytest.main(['-q', '--tb=short', 'tests/']))"
```

## CLI subcommands

| subcommand | what it produces |
|---|---|
| `audit-models` | `artifacts/audit/` — model inventory, API contract, capability matrix |
| `materialize-manifests` | `configs/benchmark/` — versioned YAML manifests |
| `smoke-long-term --dataset ETTh1` | `artifacts/smoke/` — data-centric smoke plan |
| `smoke-m4 --frequency Hourly` | `artifacts/smoke/` — M4 smoke plan |
| `probe-compatibility --models a,b` | `artifacts/review/runtime_compatibility.json` |
| `validate-canonical-forward --models a,b` | `artifacts/review/canonical_forward_validation.json` |

For `probe-compatibility` and `validate-canonical-forward`, always pass `--models <subset>` during development (probing all 29 takes minutes). Full probes are only needed before a benchmark run.

## s-transformers-lib API contract

The library uses a unified interface rooted in `BaseTransformerModel` (`s_transformers_lib/interfaces/base_model.py`). The canonical flow:

```python
from s_transformers_lib.models import create_model
from s_transformers_lib.interfaces.forecasting import ForecastInput, TrainingConfig

model = create_model("patchtst", config={...})
model.fit(train_data, val_data, training=TrainingConfig(epochs=10, device="cuda"))
output = model.predict(ForecastInput(x=x_tensor, x_mark=mark_tensor))
# output.prediction has shape (batch, pred_len, channels)
```

`ForecastInput` fields relevant to regular benchmarks: `x`, `x_mark`, `y_full`, `y_mark`. Fields `x_time`, `x_mask`, `pred_time` are only for `tpatchgnn` (excluded).

## Model eligibility

28 of 29 models are eligible. **`tpatchgnn` is permanently excluded** from the regular benchmark because it requires irregular patched inputs (`x_time`, `x_mask`, `pred_time`) that do not exist in the long-term or M4 protocols. Never reintroduce it unless a separate irregular-series track is explicitly opened.

Many models still carry `review_status: bootstrap` in the capability matrix — they were included by heuristic and have not yet had full runtime validation. When a model fails during a benchmark run, update its `eligibility_reason` and `review_status` to `manual_override` with explicit justification before marking it `N/A`.

## Invariants — never violate these

1. **Temporal split before windowing.** The scaler and window indices must not see future data.
2. **Inverse-scale before computing metrics.** All MAE, RMSE, sMAPE, MASE values must be in the original series scale.
3. **No `if model_name ==` in the runner.** Model-specific differences live in adapters, resolved from the capability matrix.
4. **No `k-fold`; use fixed temporal split + 3 seeds (42, 123, 2026).** Rolling-origin validation is supplementary only.
5. **Results regenerated from persisted artifacts only.** Never recompute a paper table from memory or a live run — always from the raw `.parquet`/`.json` files written during the run.
6. **Metrics averaged across seeds, not across datasets with different scales.** Use rank-based aggregation or per-dataset normalization for cross-dataset comparisons.

## Architecture rules

- **Discovery layer** (`src/transformer_ts_ranking/discovery/`) owns the capability matrix. Any new model enters through an audit, not by editing the runner.
- **Adapters** (`src/transformer_ts_ranking/adapters/`) translate between the loader's output and each model family's expected batch format. Default adapter handles models that satisfy the canonical contract. Specialized adapters cover families with non-standard batches.
- **Loaders** (`src/transformer_ts_ranking/data/`) are split: `long_term.py` for ETT/weather/etc., `m4.py` for M4. They share no state.
- **M4 OWA formula:** `OWA = 0.5 × (MASE_model / MASE_Naive2) + 0.5 × (sMAPE_model / sMAPE_Naive2)`. Naive2 reference from `submission-Naive2.csv`.

## Known model issues (verified against source code)

### Group A — `train_step` crashes with KeyError (missing time features)

These 8 models do bare `batch['x_mark']` / `batch['y_mark']` dict accesses with no `.get()` fallback. They crash on the first `fit()` call if the batch doesn't include time-feature tensors:

`autoformer`, `informer`, `nonstationary_transformer`, `scaleformer`, `spacetimeformer`, `tft`, `transformer (vanilla)`, `quatformer`

**Fix**: The adapter/dataloader must always inject zero time-mark tensors when real time features are unavailable.

### Group B — `predict()` raises ValueError (required forward() params with no default)

The base `_validate_predict_kwargs` enforces required params. These models require `x_mark_enc` (and some also `x_mark_dec`, `x_dec`) with no default:

- `pyraformer`, `quatformer` — need `x_mark_enc`
- `scaleformer`, `spacetimeformer`, `tft`, `transformer (vanilla)` — need `x_mark_enc`, `x_mark_dec`, `x_dec`

The base auto-builds `x_dec` from `label_len` + zeros when config has `label_len` — that part is safe. It does **not** auto-build time-mark tensors. Always pass `x_mark` and `y_mark` in `ForecastInput` for these models.

### Group C — Behavioural anomalies

| Model | Issue |
|---|---|
| `fedformer` | Default config is BERT-large scale (`hidden_size=768`, `num_hidden_layers=12`). **Will OOM on 8 GB GPU** at batch_size ≥ 4. Benchmark config must override: `hidden_size=256`, `num_hidden_layers=4`, `num_attention_heads=4`, `intermediate_size=512`. |
| `patchtst` | Default `d_model=768`, `d_ff=3072`. Tight at batch_size > 8. Benchmark config: `d_model=256`, `n_heads=8`, `d_ff=512`. |
| `lag_llama` | Autoregressive inference: runs `pred_len` sequential forward passes. At `pred_len=96`, ~96× slower than any other model. `eval_step` returns `(float, torch.Tensor)` instead of `(float, Dict)` — incompatible with any harness that reads `eval_step()[1]` as a dict. |
| `chronos2` | **No pretrained weights loaded** — trains from scratch. Fast parallel decoding (not autoregressive). Safe for the benchmark. |
| `quatformer` | Always returns `(output, reg_loss)` tuple from `forward()`. The base `_normalize_predict_output` handles this via the tuple path — safe through `predict()`. |
| `basisformer` | Returns `(output, l_entropy, l_smooth)` 3-tuple when `is_training=True`. `train_step` handles this correctly. Don't call `forward(is_training=True)` from eval code. |
| `earthformer` | Default `data_mode='auto'` is safe (selects 1D when `input_height == input_width == 1`). Always set `data_mode='1d'` explicitly in benchmark configs to prevent accidental 3D mode. |
| `airformer` | `train_step` has a `kl_weight` parameter (default 0.01) never used by the base `fit()` loop. KL-weighted stochastic training is silently bypassed. Functionally trains as a deterministic model. |
| `lag_llama` | `use_pinball_loss=True` in config overrides `loss_fn` silently. Always set `use_pinball_loss=False` in benchmark configs unless explicitly testing probabilistic track. Same for `chronos2`. |

### Adapter strategy

All fixes belong in `src/transformer_ts_ranking/adapters/`, resolved from the capability matrix. **Never put `if model_name ==` in the runner.**

Three adapter families needed:

1. **`MarkInjectorAdapter`** — wraps `train_step`/`eval_step` to inject zero `x_mark`/`y_mark` tensors when absent. Covers Group A + B.
2. **`DefaultAdapter`** — models that satisfy the full canonical contract without patching (majority).
3. **`LagLlamaAdapter`** — overrides `eval_step` to normalize the return type and flags slow inference in timing logs.

## Documentation requirements

Every module, class, and function must have a docstring. Inline comments are required where the intent is not obvious from the code. This is a scientific artifact — auditability matters.

## Paper-oriented outputs

Everything beyond raw results exists to support the publication. When implementing any reporting component, ask: "can this go directly into a paper figure or table?"

### Training & convergence analysis

Each benchmark run must persist per-epoch data for:

- Train loss and validation loss (convergence curves).
- Wall-clock time per epoch and per batch (compute budget curves).
- Peak GPU memory per epoch (VRAM usage).
- Parameter count (logged once at run start).

These feed three figure families in the paper: convergence plots, compute-cost scatter plots, and VRAM-vs-performance trade-off charts. The reporting module must be able to regenerate all figures from the persisted per-epoch JSONL without re-running any model.

### Comparative visualizations

- **Accuracy heatmaps** — model × dataset-horizon matrices for MAE and RMSE.
- **Rank distribution plots** — boxplots of per-dataset rank per model (shows consistency vs. specialization).
- **Radar/spider charts** — accuracy vs. speed vs. memory per model family.
- **Scaling curves** — metric vs. horizon for representative models per family.
- All figures must be reproducible with a single CLI call from persisted artifacts.

### Statistical validation

The paper must include statistical tests to support ranking claims:

- **Friedman test** across all models on the long-term track (non-parametric rank test for multiple comparators on multiple datasets).
- **Post-hoc Nemenyi test** (or Wilcoxon signed-rank with Bonferroni correction) to identify which pairwise differences are significant.
- **Critical Difference (CD) diagram** — standard visualization from Demšar (2006), required for any ML benchmark paper.
- Report effect sizes alongside p-values.

The `src/transformer_ts_ranking/reporting/` module will own all of this. It reads from `results/raw/results_raw.parquet` and writes to `paper/figures/` and `paper/tables/`.

### Efficiency track

Beyond accuracy, the paper profiles each model's practical cost:

- Training time per epoch (seconds).
- Inference latency per sample (milliseconds).
- Peak GPU memory (MB).
- Parameter count.
- A composite "efficiency score" = accuracy rank / (normalized compute cost) for the efficiency leaderboard.

## Artifact layout

```
artifacts/audit/          ← model_inventory.json, api_contract_report.json, model_capability_matrix.yaml
artifacts/smoke/          ← smoke plan JSONs (per dataset/frequency)
artifacts/review/         ← runtime_compatibility.json, canonical_forward_validation.json
configs/benchmark/        ← versioned YAML manifests (model_capability_matrix, datasets, presets)
results/raw/              ← results_raw.parquet (one row per run; includes per-epoch metrics)
results/leaderboards/     ← leaderboard_long_term.csv, leaderboard_short_term.csv, leaderboard_efficiency.csv
results/stats/            ← friedman_test.json, nemenyi_matrix.csv, cd_diagram_data.json
paper/tables/             ← long_term.tex, short_term.tex, coverage.tex, efficiency.tex, stats.tex
paper/figures/            ← heatmaps, convergence curves, CD diagram, radar charts (PNG + PDF)
```
