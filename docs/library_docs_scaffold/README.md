# Documentation scaffold for `s-transformers-lib`

A **portable, self-contained MkDocs + Material + mkdocstrings** documentation site, built here in the
benchmark repo (per [`docs/documentation_site_design.md`](../documentation_site_design.md)) to carry
into the **library** repository. It touches no benchmark runtime code, no `src/`, no SLURM scripts,
no `.venv`, and not the `s-transformers-lib` submodule — so it cannot affect current or future
experiment runs.

## What this contains (design-doc phases)

| File / dir | Phase | Purpose |
|---|---|---|
| `mkdocs.yml` | D1 | Material scaffold, nav (Diátaxis), plugins, math |
| `scripts/gen_ref_pages.py` | D2 | Auto **API Reference** from the source (no manual nav) |
| `scripts/gen_model_pages.py` | D5 | One page per model from `list_models()` (card + autodoc + demo) |
| `docs/index.md` | D1 | Home / landing |
| `docs/tutorials/*` | D3 | First forecast; train & evaluate |
| `docs/how-to/*` | D3 | Choose a model, add a model, normalization, dataloaders |
| `docs/explanation/*` | D3 | Unified contract, model families (with LaTeX) |
| `docs/benchmark/results.md` | D6 | Placeholder that will embed this repo's leaderboard/CD |
| `docs/cards/` | — | Where the integration layer's model cards land (auto-inlined) |
| `pyproject.docs-extra.toml` | D7/D8 | `docs` extra + `ruff D` + `interrogate` gates |
| `.readthedocs.yaml` | D7 | Read the Docs build |
| `github-workflow-docs.yml` | D7 | Alternative: GH Pages + `mike` |

Content pages (`mkdocstrings` API blocks, per-model pages) are **auto-generated** at build time; the
tutorials/how-to/explanation are real starter content grounded in the actual API
(`create_model` / `ForecastInput` / `TrainingConfig` / `create_dataloaders` / `compute_all_metrics`).

## How to adopt it in the library repo

1. Copy `mkdocs.yml`, `docs/`, and `scripts/` to the **library repo root**.
2. Merge `pyproject.docs-extra.toml` into the library's `pyproject.toml`.
3. Place `.readthedocs.yaml` at the root (or `github-workflow-docs.yml` at
   `.github/workflows/docs.yml`).
4. Verify locally:
   ```bash
   pip install ".[docs]"
   mkdocs serve      # or: mkdocs build --strict
   ```
5. Set `site_url` / `repo_url` in `mkdocs.yml`.

## Things to check on adoption

- **Package import path.** The scaffold assumes the installable package is `s_transformers_lib`
  (matching the `::: s_transformers_lib.models.<name>` autodoc directives and the code examples). If
  the library's src-layout differs, update `ROOT_PACKAGE`/`SRC_DIR` in the two `scripts/*.py` and the
  import lines in the prose pages.
- **`interrogate` `fail-under`.** Set it to the library's *current* docstring coverage first, then
  ratchet up, so the gate never blocks day one (design doc §11).
- **Notebook execution.** `mkdocs-jupyter` is set to `execute: false` (model demos may need a GPU);
  flip per-notebook if you want CI to run the light tutorial.

## Not done here (needs the library env / a completed run)

- Actually building/serving the site (needs the library installed; done in the library repo).
- **D6** results: fill `docs/benchmark/results.md` after the current benchmark run completes.
- Per-model **cards**: drop into `docs/cards/<name>.md` when the integration card generator exists;
  they are auto-inlined, with a stub fallback until then.
