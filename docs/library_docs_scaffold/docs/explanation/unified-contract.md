# The unified contract

Every model in the library, regardless of its internal architecture, is used through the same
sklearn-style flow:

```
create_model(name, config)  ->  model.fit(train, val, training)  ->  model.predict(ForecastInput)
```

This is enforced by a common base class, `BaseTransformerModel`, and typed I/O objects.

## Why a unified contract

The forecasting literature ships every model with its own training script, batch format, and
evaluation glue. Reusing or comparing them means rewriting that glue each time. A single contract
means:

- **One learning curve** for all models.
- **Swap models by name** — change one string, not a pipeline.
- **Consumers build once** — a benchmark, a product, or an agent targets the contract, not each
  model.

## The pieces

- **`create_model(name, config)`** — the registry entry point; returns a configured model.
- **`BaseTransformerModel`** — defines `fit` / `predict` and the training loop hooks.
- **`ForecastInput`** — typed inputs: `x`, `x_mark`, `y_full`, `y_mark` (plus irregular-series
  fields for the models that need them).
- **`TrainingConfig`** — epochs, device, learning rate, and related knobs.
- **`ForecastOutput`** — carries `prediction` of shape `(batch, pred_len, channels)`.

## Prediction shape

For an input window $x \in \mathbb{R}^{B \times L \times C}$ (batch $B$, sequence length $L$,
channels $C$) and horizon $H$, every model returns

$$
\hat{y} \in \mathbb{R}^{B \times H \times C}.
$$

The shape is identical across models, which is what makes them interchangeable.

See [Model families](model-families.md) for how architectures differ *under* this contract.
