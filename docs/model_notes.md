# Model Notes for Paper

Technical notes on all 29 models in s-transformers-lib, covering benchmark
configuration, known issues, applied fixes, and behaviours relevant to the paper.
These notes feed the paper's appendix on experimental setup and model descriptions.

---

## Eligibility summary

| Status | Count | Models |
|---|---|---|
| Eligible (long-term + M4) | 28 | all except `tpatchgnn` |
| Permanently excluded | 1 | `tpatchgnn` |

---

## Permanently excluded models

### `tpatchgnn`

**Reason:** Requires irregular-series inputs (`x_time`, `x_mask`, `pred_time`) that
do not exist in either the long-term or M4 protocols.  Including it would require a
separate irregular-series benchmark track that is outside the scope of this paper.
No workaround is possible without fundamentally changing the benchmark protocol.

---

## Benchmark configuration overrides

All overrides are applied in `src/transformer_ts_ranking/benchmark/model_configs.py`
through the `_MODEL_OVERRIDES` and `_M4_MODEL_OVERRIDES` dicts.  No `if model_name ==`
branches exist in the runner or engine — all model-specific differences are
data-driven from these dicts and the two accessor functions
`get_batch_size_override()` and `get_context_len_override()`.

---

## Per-model notes

### `airformer`

- **Family:** Encoder-only (stochastic latent).
- **Anomaly:** `train_step` accepts a `kl_weight` parameter (default 0.01) for
  KL-weighted stochastic training, but the base `fit()` loop never passes it.
  The model effectively trains as a deterministic model — the KL term is silently
  bypassed.  This is noted in the paper but not fixed: the deterministic behaviour
  is consistent across all seeds and the model still learns a useful representation.
- **Config:** Uses benchmark defaults (`d_model=128, d_ff=256, n_heads=4, e_layers=2`).

---

### `autoformer`

- **Family:** Seq2seq (series decomposition + autocorrelation attention).
- **Group A (time-mark injection):** `train_step` does `batch['x_mark']` with no
  `.get()` fallback — crashes without time marks.  The benchmark's `LongTermWindowDataset`
  always injects zero time-mark tensors (`x_mark`, `y_mark`) so this is handled
  transparently.
- **Config:** Uses `label_len` from the dataset manifest (seq2seq family).
- **Known device bug (fixed):** Early runs on V100 produced `RuntimeError: Expected all
  tensors to be on the same device` because the `y_full` tensor was not moved to the
  correct device inside `train_step`.  Fixed in `_MODEL_OVERRIDES` by ensuring `y_full`
  is always passed as a CPU fallback when not needed.

---

### `basisformer`

- **Family:** Seq2seq (learnable basis functions for trend/seasonality decomposition).
- **Anomaly:** `forward(is_training=True)` returns a 3-tuple
  `(output, l_entropy, l_smooth)`.  `train_step` handles this correctly.  Calling
  `forward(is_training=True)` from eval code would break — the engine always calls
  `model.predict()` which routes through the base `_normalize_predict_output`, safe.
- **Config:** Uses `label_len` (seq2seq family).

---

### `cats`

- **Family:** Seq2seq (cross-variable aggregation transformer).
- **Config:** Uses `label_len` (seq2seq family).  Uses benchmark defaults otherwise.

---

### `chronos2` (Chronos-Bolt)

- **Family:** Pretrained zero-shot foundation model (T5-based).
- **Important:** Despite the name, **no pretrained weights are loaded** in the
  current s-transformers-lib wrapper — the model trains from scratch using the
  Chronos-Bolt architecture.  The benchmark therefore treats it as a
  high-capacity seq2seq model, not a zero-shot baseline.
- **Config override:** `model_id="amazon/chronos-bolt-small"`, `torch_dtype="float32"`.
  The `-small` variant (~21 M params) is used for VRAM balance.  Switch to
  `"amazon/chronos-bolt-base"` (~205 M) for best accuracy at higher compute cost.
- **Inference:** Parallel (non-autoregressive) decoding — fast.

---

### `chronos_bolt` (pretrained zero-shot)

- **Family:** Pretrained zero-shot foundation model.
- **Config override:** `model_id="amazon/chronos-bolt-small"`, `torch_dtype="float32"`.
- **Zero-shot:** `fit()` is a no-op; all computation is in `predict()`.  The engine
  injects the dataset scaler via `model._benchmark_scaler` so the StandardScaler
  round-trip is handled inside the wrapper (the pretrained model expects original-scale
  values).
- **Inference:** Parallel decoding; fast.

---

### `contiformer`

- **Family:** Encoder-only (continuous-time ODE attention).
- **Core issue (OOM):** `ODELinear.forward()` computes pairwise ODE integrals between
  all T encoder time steps by allocating `[B, T, T, D]` intermediate tensors (where
  D = head dimension, T = sequence length).  This is O(B × T² × D) per attention layer.
  At the benchmark's standard T=336, B=16, the ODE tensors alone exceed 11 GB.
- **Fix (context truncation):** The engine truncates the encoder input to the last
  96 time steps before every `fit()` and `predict()` call via `_TruncatedLoader`
  and inline slicing, resolved from `_MODEL_CONTEXT_LEN["contiformer"] = 96`.
  At T=96 the ODE tensor is ~150 MB — well within budget.
- **Scientific note for paper:** ContiFormer's continuous-time ODE design allows it to
  operate on any context length; 96 steps is the benchmark's chosen budget given the
  GPU constraints.  The paper reports this as a scalability limitation of the
  ODE-pairwise-integral mechanism and notes the 96-step context explicitly in the
  experimental setup table.
- **bf16 benefit:** On A100 nodes the bf16 autocast halves the ODE tensor memory
  further (~75 MB at T=96).
- **M4:** M4 sequences use `seq_len=96` by default, so the context truncation is a no-op.
- **Config:** Uses benchmark encoder-only defaults.

---

### `crossformer`

- **Family:** Encoder-only (cross-dimension segment-based attention).
- **Config:** Uses benchmark defaults.

---

### `deformable_tst`

- **Family:** Seq2seq (deformable attention variant of PatchTST).
- **Config:** Uses `label_len` (seq2seq family).

---

### `earthformer`

- **Family:** Encoder-only (spatiotemporal cuboid attention).
- **Anomaly:** Default `data_mode='auto'` selects 3D image mode when
  `input_height != 1` or `input_width != 1`.  For time-series data (1D),
  this silently switches to 3D mode causing shape mismatches.
- **Config override:** `data_mode='1d'` is always set explicitly in
  `_MODEL_OVERRIDES` to prevent accidental 3D mode.

---

### `etsformer`

- **Family:** Encoder-only (exponential smoothing transformer).
- **Anomaly:** `ETSFormerConfig` enforces `assert e_layers == d_layers`.  The base
  config sets `d_layers=1` (encoder-only default), violating this for `e_layers=2`.
- **Config override:** `d_layers=2` set in `_MODEL_OVERRIDES` to match `e_layers`.

---

### `fedformer`

- **Family:** Seq2seq (frequency-enhanced decomposed transformer).
- **OOM risk:** Default config is BERT-large scale (`hidden_size=768`,
  `num_hidden_layers=12`).  This OOMs on 8 GB GPUs at `batch_size ≥ 4`.
- **Config override:** `hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
  intermediate_size=512, d_model=256, d_ff=512, n_heads=4`.
- **Config:** Uses `label_len` (seq2seq family).

---

### `filmformer`

- **Family:** Encoder-only (FiLM: frequency-improved legendre memory).
- **Config:** Uses benchmark defaults.

---

### `informer`

- **Family:** Seq2seq (ProbSparse self-attention).
- **Group A (time-mark injection):** Same as `autoformer` — requires `x_mark`/`y_mark`.
  Handled by the benchmark's zero time-mark injection.
- **Config:** Uses `label_len` (seq2seq family).

---

### `itransformer`

- **Family:** Encoder-only (inverted embedding: variables as tokens).
- **Config:** Uses benchmark defaults.  Strong baseline on multivariate datasets.

---

### `lag_llama`

- **Family:** Encoder-only (autoregressive, pretrained on Lag-Llama architecture).
- **Inference:** Autoregressive — runs `pred_len` sequential forward passes.  At
  `pred_len=96`, approximately 96× slower than parallel models.  At `pred_len=720`,
  720× slower.  Isolated in dedicated SBATCH jobs to avoid blocking other experiments.
- **Anomaly (return type):** `eval_step()` returns `(float, torch.Tensor)` instead
  of the standard `(float, Dict)`.  Any harness that reads `eval_step()[1]` as a
  dict will break.  The engine does not call `eval_step()` directly (uses `fit()`
  + `predict()`), so this is safe in the current benchmark.
- **Config override:** `use_pinball_loss=False` to disable the pinball/quantile loss
  and use deterministic MSE for a fair comparison.
- **Paper note:** Report per-horizon inference latency separately for this model;
  the ~pred_len× slowdown is a fundamental property of autoregressive generation.

---

### `lag_llama_pretrained`

- **Family:** Pretrained zero-shot foundation model (autoregressive Lag-Llama).
- **Pretrained source:** HuggingFace `time-series-foundation-models/Lag-Llama`,
  file `model.safetensors`.  The s-transformers-lib wrapper downloads and loads these
  weights; `fit()` is a no-op.
- **State-dict mismatch (fixed 2026-06-XX):** The base config builder injected
  generic defaults `d_ff=256, n_heads=4` into the config dict.  `filter_config_for_model()`
  passed these to `LagLlamaPretrainedConfig.__init__()`, overriding the dataclass
  field defaults (`d_ff=512, n_heads=8`).  The backbone was built with `d_ff=256`,
  producing `gate_proj.weight` shape `(256, 144)` vs the checkpoint's `(512, 144)`.
  **Fix:** `_MODEL_OVERRIDES["lag_llama_pretrained"]` now explicitly sets
  `d_ff=512, n_heads=8, n_layers=8` (all architecture params that the checkpoint
  requires).  Verified: `missing_keys=[], unexpected_keys=[]`.
- **Correct architecture:** `d_model=144, d_ff=512, n_heads=8, n_layers=8, input_dim=92`.
  The `embed_inputs.weight` shape `(144, 92)` confirms `input_dim=92` (92 Lag features).
- **`_build_lag_features` shape bug (patched at runtime):** The library method calls
  `loc.expand(B, T, 1)` where `loc` has shape `(B, 1)` (2-D).  PyTorch auto-prepends a
  singleton when the target rank is higher, turning `(B, 1)` into `(1, B, 1)`.  Expanding
  dim-1 from B to T then fails: `"expanded size (T) must match existing size (B)"` whenever
  `batch_size ≠ seq_len` (always true in the benchmark: B=16, T=96 or 336).
  **Fix:** instance-level monkey-patch in `_MODEL_PATCHES["lag_llama_pretrained"]`
  (model_configs.py).  Inserts `.unsqueeze(1)` before the expand: `(B,1)→(B,1,1)→(B,T,1)`.
  Applied by `patch_model_instance()` in the engine right after `create_model()`.
  Library source is NOT modified.
- **Inference:** Autoregressive — same slowdown as `lag_llama`.  Dedicated SBATCH jobs.
- **Config override:** `hf_repo`, `hf_filename`, `scaling="mean"`, plus all architecture
  params explicitly (see above).

---

### `micn`

- **Family:** Encoder-only (Multi-scale Isometric Convolution Network).
- **Config:** Uses benchmark defaults.

---

### `multipatchformer`

- **Family:** Encoder-only (multi-scale patch transformer).
- **M4 override:** `n_sar_steps=4` (default 8 requires `pred_len ≥ 8`;
  M4 Yearly has `pred_len=6`).  Set in `_M4_MODEL_OVERRIDES`.

---

### `nonstationary_transformer`

- **Family:** Seq2seq (non-stationary normalization + de-stationary attention).
- **Group A (time-mark injection):** Same as `autoformer`.
- **Config:** Uses `label_len` (seq2seq family).

---

### `patchtst`

- **Family:** Seq2seq (patch-based channel-independent transformer).
- **OOM risk:** Default `d_model=768, d_ff=3072` is tight at `batch_size > 8`.
- **Config override:** `d_model=256, n_heads=8, d_ff=512`.
- **Config:** Uses `label_len` (seq2seq family).

---

### `pyraformer`

- **Family:** Encoder-only (pyramidal attention with hierarchical aggregation).
- **Group B (predict kwargs):** `predict()` raises `ValueError` for missing
  `x_mark_enc` with no default.  The benchmark's `ForecastInput` always passes
  `x_mark`, which the base `_validate_predict_kwargs` maps to `x_mark_enc`.
  Handled transparently.

---

### `quatformer`

- **Family:** Seq2seq (quaternion self-attention).
- **Group A (time-mark injection):** Same as `autoformer`.
- **Group B (predict kwargs):** Requires `x_mark_enc`.  Handled via `ForecastInput.x_mark`.
- **Anomaly (return tuple):** `forward()` always returns `(output, reg_loss)` 2-tuple.
  The base `_normalize_predict_output` handles this via the tuple path — safe through
  `predict()`.
- **Config:** Uses `label_len` (seq2seq family).

---

### `reformer`

- **Family:** Seq2seq (LSH attention for O(T log T) complexity).
- **Config:** Uses `label_len` (seq2seq family).  Uses benchmark defaults.

---

### `scaleformer`

- **Family:** Seq2seq (iterative multi-scale transformer).
- **Assertion:** Default `scales=[8,4,2,1]` requires `pred_len % max(scales) == 0`.
  Long-term illness horizons [24, 36, 48, 60] violate this (36 % 8 ≠ 0, 60 % 8 ≠ 0).
- **Config override:** `scales=[1]` in `_MODEL_OVERRIDES` (applies to both long-term
  and M4) disables multi-scale but makes the model compatible with any `pred_len`.
  This is a necessary simplification; the paper notes it as a benchmark constraint.
- **Config:** Uses `label_len` (seq2seq family).

---

### `spacetimeformer`

- **Family:** Seq2seq (factorized space-time attention over all variables and time).
- **Dual quadratic memory:** The decoder has TWO quadratic attention terms:
  - Temporal: O(B × C × H × T_dec²) — grows with horizon squared.
  - Spatial:  O(B × T_dec × H × C²) — grows with channel count squared.
  At B=16, C=7, h=720 (T_dec=768) the temporal term alone is ~8.5 GB.
- **Global batch-size override — B=4 for all horizons:**
  `_MODEL_BATCH_OVERRIDES["spacetimeformer"] = 4` (flat int, applies at all horizons).
  At B=4, small-channel datasets (C≤21) stay under 3.5 GB total.
  For electricity (C=321): h=96 → 5.5 GB ✓, h=192 → 11.1 GB ✓, h=336 → 22.2 GB ✓,
  h=720 → 68.7 GB ✗ (OOM, error row, N/A in paper).
  For traffic (C=862): spatial term alone exceeds 40 GB at ALL horizons even at B=1 —
  all four traffic rows are N/A for spacetimeformer.
- **Positional embedding fix (applied):** Default `max_seq_len=512` causes a CUDA
  index out-of-bounds for h=720 (decoder indexed up to T_dec=768).
  `_MODEL_OVERRIDES["spacetimeformer"]["max_seq_len"] = 1024` prevents this.
- **Config:** Uses `label_len` (seq2seq family).

---

### `tft` (Temporal Fusion Transformer)

- **Family:** Seq2seq (gated residual networks with multi-head attention).
- **Group A (time-mark injection):** Same as `autoformer`.
- **Group B (predict kwargs):** Requires `x_mark_enc`, `x_mark_dec`, `x_dec`.
  The base class auto-builds `x_dec` from `label_len` + zeros when `label_len > 0`.
  Always pass `x_mark` and `y_mark` in `ForecastInput`.
- **Config:** Uses `label_len` (seq2seq family).  **`tft` is explicitly listed in
  `_SEQ2SEQ_MODELS`.**  Without this, `label_len=0` is passed and Python's
  `batch_y[:, -0:, :]` (since −0 == 0) returns the full encoder sequence, silently
  corrupting the decoder context.  The TFT source accesses `label_len` in at least
  10 places (`batch_y[:, :self.label_len, :]`, `x_mark_dec[:, c.label_len:, :]`, etc.).

---

### `transformer` (vanilla)

- **Family:** Seq2seq (original Vaswani 2017 encoder-decoder).
- **Group A (time-mark injection):** Same as `autoformer`.
- **Group B (predict kwargs):** Requires `x_mark_enc`, `x_mark_dec`, `x_dec`.
- **Config:** Uses `label_len` (seq2seq family).

---

### `tsmixer`

- **Family:** Encoder-only (MLP-Mixer for time series).
- **Config:** Uses benchmark defaults.

---

### `timesnet`

- **Family:** Encoder-only (2D convolution via FFT-based period detection).
- **Config:** Uses benchmark defaults.

---

### `timemixer`

- **Family:** Encoder-only (multi-resolution mixing).
- **Config:** Uses benchmark defaults.

---

## Cross-cutting notes for paper sections

### Precision

All models run in **float32** by default (the engine calls `.float()` on input tensors
before every `fit()` and `predict()` call).  On nodes with A100 GPUs that support bf16
natively (`torch.cuda.is_bf16_supported() == True`), the engine applies
`torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` as a context manager
around every `model.fit()` and `model.predict()` call for **bf16-compatible models**.
This halves peak activation memory and speeds up matrix multiplications via bf16 tensor
cores, while keeping parameters in fp32 (standard AMP behaviour).

**bf16-incompatible models (run in fp32 even on A100):** `airformer`, `autoformer`,
`crossformer`, `earthformer`, `etsformer`, `fedformer`, `pathformer`.  These use
`torch.fft.*` (AutoCorrelation, ETSFormer, FEDformer, Pathformer), stochastic latent
distributions (`airformer`), or custom attention kernels (`crossformer`, `earthformer`)
that raise `TypeError: Got unsupported ScalarType BFloat16` or `RuntimeError:
Unsupported dtype BFloat16` under autocast.  Excluded via `_MODEL_NO_BF16` in
`model_configs.py`.

On V100 nodes (`is_bf16_supported() == False`), all models run in fp32.

### Seq2seq vs encoder-only split

Models in the seq2seq family use a `label_len > 0` decoder context (a slice of the
encoder history prepended to the decoder input).  For long-term benchmarks,
`label_len` is read from the dataset manifest (typically 48 for ETT datasets, 0 for
exchange_rate/illness).  For M4, `label_len = min(horizon // 2, 24)`.  All other
models use `label_len=0` (encoder-only).

### Time-feature injection (Group A)

Eight models (`autoformer`, `informer`, `nonstationary_transformer`, `scaleformer`,
`spacetimeformer`, `tft`, `transformer`, `quatformer`) access `batch['x_mark']` and
`batch['y_mark']` directly with no `.get()` fallback in their `train_step`.  The
benchmark's `LongTermWindowDataset` and `M4SeriesDataset` always include these keys
(zero tensors if no real time features are needed), so these models are safe without
any special handling in the engine.

### Foundation model baselines

`chronos_bolt` (pretrained) and `lag_llama_pretrained` are zero-shot models where
`fit()` is a no-op.  They appear in both accuracy tables and in a separate
"foundation model" column to allow fair comparison: one row shows their zero-shot
performance, a second note clarifies they were not trained on the benchmark datasets.

### M4 OWA formula

OWA = 0.5 × (MASE_model / MASE_Naive2) + 0.5 × (sMAPE_model / sMAPE_Naive2)

Naive2 reference values are taken from `data/m4/submission-Naive2.csv`.  All M4
metrics are computed in the original scale after per-series z-score denormalisation.

### Efficiency track caveats

- `lag_llama` and `lag_llama_pretrained`: inference latency scales linearly with
  `pred_len` (autoregressive).  Their latency entries in the efficiency leaderboard
  are reported at each horizon separately rather than as a single per-sample value.
- `contiformer`: inference latency scales with T² (ODE integral evaluation).
  At context_len=96 it is comparable to other encoder-only models.
- `spacetimeformer`: training time scales with C² (spatial attention).  For
  high-channel datasets, training is significantly slower than single-variable or
  low-channel models.

---

*Last updated: 2026-06-24.  All fixes applied in commits on the main branch.*
