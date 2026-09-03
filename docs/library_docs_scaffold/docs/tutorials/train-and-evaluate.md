# Train & evaluate a model end-to-end

This tutorial builds dataloaders from a raw array, trains a model, and computes accuracy metrics —
all with library utilities.

## 1. Build dataloaders

`create_dataloaders` performs the **temporal split before windowing** (no leakage) and returns
train / val / test loaders:

```python
from s_transformers_lib.data.datasets import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data,                 # (total_len, n_features)
    seq_len=96,
    pred_len=96,
    label_len=48,         # decoder context for seq2seq families; 0 for encoder-only
    mode="encoder_only",  # or "seq2seq"
    batch_size=32,
)
```

## 2. Train

```python
from s_transformers_lib.models import create_model
from s_transformers_lib.interfaces.forecasting import TrainingConfig

model = create_model("itransformer")
model.fit(train_loader, val_loader, training=TrainingConfig(epochs=20, device="cuda"))
```

## 3. Evaluate

Use the built-in metrics (all in original scale — inverse-transform before computing them if you
scaled the inputs):

```python
from s_transformers_lib.data.metrics import compute_all_metrics

# preds, targets: (n, pred_len, channels)
metrics = compute_all_metrics(preds, targets)   # mse, mae, rmse, mape, smape
print(metrics)
```

!!! warning "Scale matters"
    Compute metrics on **inverse-scaled** values so errors are in the original series units. See
    [Normalization](../how-to/normalization.md).

## Next

- [Build dataloaders](../how-to/dataloaders.md) — model-aware batch shapes.
- [Benchmark & Results](../benchmark/results.md) — how models compare across datasets.
