# Your first forecast

This tutorial produces a forecast with a trained model in a few lines. It assumes you have a
multivariate series as a tensor of shape `(batch, seq_len, channels)`.

## 1. Create a model

Every model is built from the registry by name and an optional config dict:

```python
from s_transformers_lib.models import create_model, list_models

print(list_models())          # every registered model
model = create_model("patchtst", config={"d_model": 256, "n_heads": 8, "d_ff": 512})
```

## 2. Train it

Training goes through the unified `fit()` API with a `TrainingConfig`:

```python
from s_transformers_lib.interfaces.forecasting import TrainingConfig

model.fit(train_data, val_data, training=TrainingConfig(epochs=10, device="cuda"))
```

## 3. Predict

Predictions use a typed `ForecastInput`. The output's `prediction` has shape
`(batch, pred_len, channels)`:

```python
from s_transformers_lib.interfaces.forecasting import ForecastInput

output = model.predict(ForecastInput(x=x_tensor, x_mark=mark_tensor))
y_hat = output.prediction     # (batch, pred_len, channels)
```

!!! tip "Time features"
    Some model families require calendar/time-mark tensors (`x_mark`, `y_mark`). If unsure, pass
    them — see [Choose a model](../how-to/choose-a-model.md) and the model's page for its
    requirements.

## Next

- [Train & evaluate](train-and-evaluate.md) — a complete loop with metrics.
- [Choose a model](../how-to/choose-a-model.md) — pick the right model for your data.
