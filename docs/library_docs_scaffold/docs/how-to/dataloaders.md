# How to build dataloaders

Use `create_dataloaders` to turn a raw array into train / val / test loaders with a leakage-free
temporal split.

```python
from s_transformers_lib.data.datasets import create_dataloaders

train, val, test = create_dataloaders(
    data,                 # (total_len, n_features)
    seq_len=96,
    pred_len=96,
    label_len=48,         # decoder context length (seq2seq); 0 for encoder-only
    stride=1,
    time_stamps=stamps,   # optional; enables time-feature marks
    mode="encoder_only",  # or "seq2seq"
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=32,
)
```

## Choosing `mode` and `label_len`

These depend on the model **family**:

- **Encoder-only** models: `mode="encoder_only"`, `label_len=0`.
- **Seq2seq** models: `mode="seq2seq"`, `label_len>0` (a slice of the encoder history prepended to
  the decoder input).

!!! tip "Model-aware loaders (planned)"
    The integration design adds `build_dataloaders(model_name, data, ...)` that resolves `mode`,
    `label_len`, and time-mark injection from the model's declared capabilities — so you won't need
    to know each model's batch quirks. Until then, set them from the model's page. See the
    integration design doc, §9.

## Custom datasets

For full control, use `TimeSeriesDataset` directly and wrap it in a PyTorch `DataLoader`. See the
[API Reference](../reference/).
