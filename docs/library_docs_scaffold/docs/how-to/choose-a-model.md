# How to choose a model

The library exposes 20+ models. Pick one by matching your data and task to a model's capabilities.

## By data shape

| Your data | Consider |
|---|---|
| Regular multivariate series, long horizon | `patchtst`, `itransformer`, `cats`, `timexer` |
| Channel-independent / many channels | `patchtst`, `multipatchformer` |
| Needs calendar/time features | encoder-decoder families (`autoformer`, `informer`, `fedformer`, …) |
| Zero-shot / pretrained | `chronos2`, `lag_llama` |
| Irregular sampling | models declaring `requires_irregular_times` |

## Programmatically

Filter the registry by capability (once the capability API is available):

```python
from s_transformers_lib import list_models
# list_models(filter=lambda c: c.supports_regular_mts and not c.requires_irregular_times)
print(list_models())
```

Each model's page documents its **family**, required inputs, and known constraints — start there,
then consult the [Benchmark & Results](../benchmark/results.md) for empirical rankings.

!!! note
    Selection guidance becomes machine-readable through the model cards and the `recommend_models`
    tool described in the integration design. Until then, use this table plus the benchmark ranking.
