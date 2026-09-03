# s-transformers-lib

A unified toolkit of **state-of-the-art transformer models for time-series forecasting**, exposed
through a single, consistent API.

Every model — from classic encoder-decoder transformers to continuous-time ODE attention and
pretrained foundation models — is created, trained, and queried the same way:

```python
from s_transformers_lib.models import create_model
from s_transformers_lib.interfaces.forecasting import ForecastInput, TrainingConfig

model = create_model("patchtst", config={"d_model": 256, "n_heads": 8, "d_ff": 512})
model.fit(train_data, val_data, training=TrainingConfig(epochs=10, device="cuda"))

output = model.predict(ForecastInput(x=x_tensor, x_mark=mark_tensor))
output.prediction  # (batch, pred_len, channels)
```

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Tutorials](tutorials/first-forecast.md)** — learn by doing: your first
  forecast, then a full train-and-evaluate loop.
- :material-tools: **[How-to guides](how-to/choose-a-model.md)** — accomplish a task: choose a model,
  add a new one, use normalization, build dataloaders.
- :material-book-open-variant: **[Explanation](explanation/unified-contract.md)** — understand the
  design: the unified contract and the model families.
- :material-api: **[API Reference](reference/)** — every public symbol, generated from the source.
- :material-cube-outline: **[Models](models/)** — one page per model: card, API, and demo.
- :material-chart-box: **[Benchmark & Results](benchmark/results.md)** — how the models rank on
  long-term and M4 forecasting.

</div>

## Install

```bash
pip install s-transformers-lib
```

Optional extras: `s-transformers-lib[docs]` (build these docs), `s-transformers-lib[agent]`
(agent/MCP interface).
