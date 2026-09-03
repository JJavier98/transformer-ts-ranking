# How to normalize inputs (RevIN & scalers)

Transformer forecasters are sensitive to input scale. The library ships two complementary tools in
`s_transformers_lib.data`.

## Scalers

Standardize features before windowing. **Fit on the training split only**, then apply to val/test,
so no future statistics leak:

```python
from s_transformers_lib.data.scalers import StandardScaler   # see API reference for exact names

scaler = StandardScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
```

Always **inverse-transform predictions and targets** before computing metrics, so errors are in the
original units.

## RevIN

Reversible Instance Normalization normalizes each instance on the way in and denormalizes on the way
out — useful for distribution shift. It is available as `s_transformers_lib.data.revin` and is used
internally by several models.

!!! warning "Order of operations"
    Split → fit scaler on train → window → model (optional RevIN) → predict → inverse-scale →
    metrics. Fitting the scaler after windowing, or on the whole series, leaks information.

See the [API Reference](../reference/) for the exact class names and signatures.
