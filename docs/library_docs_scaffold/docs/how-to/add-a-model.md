# How to add a new model

New models follow the library's standard structure and enter the registry so `create_model` and the
docs pick them up automatically.

## 1. Create the model package

```
src/models/<model_name>/
  __init__.py
  config.py      # dataclass config, inheriting the base config
  attention.py   # attention variant(s)
  blocks.py      # building blocks
  model.py       # the model, subclassing BaseTransformerModel
```

Subclass `BaseTransformerModel` and honor the unified contract (`fit` / `predict`,
`ForecastInput` / `ForecastOutput`). See [The unified contract](../explanation/unified-contract.md).

## 2. Register it

Add the model to the registry so it appears in `list_models()`.

## 3. Tests (TDD)

Add `tests/test_<model_name>.py` covering initialization from config, a forward pass without errors,
and strict input/output shape checks.

## 4. Demo notebook

Add `examples/<model_name>_demo.ipynb` — import the model, move it to GPU if available, run a
commented forward pass, and show basic metrics. The docs embed this notebook automatically on the
model's page.

## 5. Documentation

The model's docs page is **auto-generated** from `list_models()` + its docstrings — no manual page
needed. Keep the docstrings NumPy-style and complete. Update the `README` model listing (kept in
alphabetical order).

!!! tip "It appears in the docs for free"
    Because the API reference and model pages are generated from the source, a well-documented,
    registered model shows up in the site with no navigation edits.
