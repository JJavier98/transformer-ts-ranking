# Design Document — Documentation Site for `s-transformers-lib` (MkDocs + Material + mkdocstrings)

| | |
|---|---|
| **Status** | Draft / RFC — for review in the `s-transformers-lib` repository |
| **Author** | J. Javier A. R. |
| **Date** | 2026-09-03 |
| **Target repo** | `s-transformers-lib` (the library), **not** `transformer-ts-ranking` (this benchmark) |
| **Decided stack** | **MkDocs + Material theme + mkdocstrings** (structure: **Diátaxis**); hosting: Read the Docs |
| **Related** | `docs/agentic_integration_design.md` (shares the docstring/model-card source), `docs/model_notes.md` |

> Produced *from* the benchmark repo as a portable deliverable; the work belongs in the library.
> Nothing here is implemented yet, and the benchmark must not modify the submodule.
>
> **Relationship to the integration RFC.** This is a **separate, independent** effort from the
> agentic integration layer. They meet at exactly one point: both the human **API Reference** and the
> agent **model cards** derive from the *same source* (docstrings + `ModelCapabilities`). That shared
> source is a reason to avoid duplicating text, **not** a build-order dependency — see §10.

---

## 1. Motivation

The library has a clean API and per-model demo notebooks, but no **rendered, navigable documentation
site**. For a toolkit meant to be *marketed to the time-series community*, that site is the front
door: it is what makes the library discoverable, learnable, and trustworthy to a newcomer. The goal
is a professional docs site, **built from existing material** (docstrings, README, example
notebooks) with minimal new prose, that stays current automatically.

### Goals

- A published, versioned documentation site with an **auto-generated API reference**.
- **Docs-as-code:** everything lives in the repo, builds in CI, and is reviewed like code.
- **Single source of truth:** API docs come from docstrings; no hand-copied signatures.
- Render the existing **example notebooks** as tutorials without rewriting them.
- Structure that scales to 28+ models without becoming a wall of text (Diátaxis).

### Non-goals

- No rewrite of the models or their docstrings' *content* (only their rendering).
- No benchmark/ranking logic (the results section *embeds* this repo's output; it is not computed in
  the library).
- Not the agentic layer — model cards as a machine artifact are the integration RFC's concern; here
  they are only *rendered* for humans (§10).

---

## 2. Chosen standard

Two layers, both current-standard:

- **Structure — Diátaxis.** The widely adopted framework (Django, NumPy, many): four documentation
  modes that are never mixed — *Tutorials, How-to guides, Reference, Explanation*.
- **Tooling — MkDocs + Material + mkdocstrings.** Markdown-based, excellent default UX, and
  `mkdocstrings` reads the library's **NumPy-style** docstrings directly to auto-generate the API
  reference. Chosen over Sphinx for lower friction and community-facing polish; Sphinx remains a
  valid alternative but is not used here.

Foundation for both: **docstrings**. The library already uses NumPy style; this effort enforces and
renders them, it does not invent them.

---

## 3. Design principles

1. **Single source of truth.** API pages are generated from docstrings via mkdocstrings — never
   hand-maintained. Model descriptions come from one place shared with the integration model cards.
2. **Build-from-existing.** Reuse docstrings, `README`, and `examples/*.ipynb`; write only the glue
   (nav, tutorials index, explanation pages).
3. **Docs-as-code.** Docs live beside the code, build in CI on every PR, and fail the build on
   broken references or (optionally) docstring-coverage regressions.
4. **Additive & decoupled.** Adding the docs site changes no runtime code and no model behavior.
5. **Structured by Diátaxis.** Every page has exactly one mode; the nav mirrors it.

---

## 4. Structure — Diátaxis mapped to this library

| Mode | Purpose | Pages here |
|---|---|---|
| **Tutorials** | Learn by doing | "First forecast in 10 lines"; "Train & evaluate a model end-to-end" |
| **How-to guides** | Accomplish a task | "Add a new model", "Use RevIN / scalers", "Build model-specific dataloaders", "Choose a model for my data" |
| **Reference** | Describe the API | Auto-generated: `create_model`, `list_models`, `ForecastInput`, `TrainingConfig`, `BaseTransformerModel`, `data/*`, `metrics`; one page per model |
| **Explanation** | Understand the why | "The unified `config → fit → predict` contract", "Model families (encoder-only vs seq2seq)", "Normalization: RevIN & scalers", design philosophy |

Plus a **Benchmark & Results** section that embeds this repo's leaderboard + CD diagram (§10), and a
**Home** landing page.

---

## 5. Toolchain components

### 5.1 Docstrings (foundation)
- Keep **NumPy style**. Enforce with `ruff` `D` (pydocstyle) rules; measure with **`interrogate`**
  (CI gate at an agreed coverage %, e.g. 90%).

### 5.2 Site generator — MkDocs + Material
- `mkdocs` + `mkdocs-material`. Material gives navigation, search, light/dark, and code copy out of
  the box.

### 5.3 API reference — mkdocstrings (Python handler)
- `mkdocstrings[python]` (Griffe backend), `docstring_style: numpy`.
- Auto-build the reference page tree from the package with **`mkdocs-gen-files`** +
  **`mkdocs-literate-nav`** (the standard mkdocstrings recipe): a small `gen_ref_pages.py` script
  walks `src/` and emits one stub per module, so new modules/models appear in the API section with
  no manual nav edits.

### 5.4 Notebooks as tutorials — mkdocs-jupyter
- `mkdocs-jupyter` renders `examples/<model>_demo.ipynb` directly. Decide **execute-in-CI**
  (freshness, slower/needs GPU-free path) vs **render-as-saved** (fast, but authors must re-run).
  Recommendation: render-as-saved for model demos (they may need GPU), execute-in-CI only for the
  lightweight "first forecast" tutorial.

### 5.5 Math — arithmatex
- `pymdownx.arithmatex` + MathJax for the transformer equations in explanation pages.

### 5.6 Versioning + hosting
- **`mike`** for versioned docs (a version selector: `latest`, `stable`, per release).
- **Read the Docs** (native MkDocs support, build-per-commit) or GitHub Pages via
  `mkdocs gh-deploy` in Actions. RTD recommended for the version matrix and OSS defaults.

---

## 6. Repository layout (in the library)

```
s-transformers-lib/
  mkdocs.yml
  docs/
    index.md                     # Home / landing
    tutorials/
      first-forecast.md
      train-and-evaluate.md
    how-to/
      add-a-model.md
      normalization.md
      dataloaders.md
      choose-a-model.md
    explanation/
      unified-contract.md
      model-families.md
    models/                      # one page per model (card + autodoc + demo)
      patchtst.md
      ...
    reference/                   # AUTO-GENERATED (gen_ref_pages.py) — not hand-written
    benchmark/
      results.md                 # embeds transformer-ts-ranking outputs (§10)
  scripts/gen_ref_pages.py       # mkdocstrings recipe: build reference tree
  pyproject.toml                 # [project.optional-dependencies] docs = [...]
  .readthedocs.yaml              # or .github/workflows/docs.yml for GH Pages
```

`docs` dependencies go in an **optional extra** (`pip install s-transformers-lib[docs]`) so they do
not weigh on normal installs.

---

## 7. What each page for a model contains

To avoid a thin, repetitive "Reference dump", each `models/<name>.md` composes three blocks from
**one source each** — no duplication:

1. **Model card** (rendered) — summary, family, paper, capabilities, selection hints. *Source:* the
   integration RFC's model card (§10) when it exists; until then, a short front-matter block.
2. **API reference** (autodoc) — `::: s_transformers_lib.models.<name>` via mkdocstrings.
3. **Demo** — the embedded `examples/<name>_demo.ipynb`.

---

## 8. Phasing

| Phase | Deliverable | Depends on |
|---|---|---|
| **D1** | `mkdocs.yml` + Material scaffold + Home; builds locally | — |
| **D2** | Auto API reference (mkdocstrings + gen_ref_pages + literate-nav) | D1 |
| **D3** | Tutorials + how-to guides (incl. "add a model", "choose a model") | D1 |
| **D4** | Embed example notebooks (mkdocs-jupyter) | D1 |
| **D5** | Per-model pages (card block + autodoc + demo) | D2, D4 |
| **D6** | Benchmark & Results section (embeds this repo's leaderboard/CD) | a completed benchmark run |
| **D7** | Versioning (`mike`) + hosting (Read the Docs) + CI build gate | D2 |
| **D8** | Docstring-coverage gate (`interrogate`) + `ruff D` in CI | — |

D1–D5 and D7–D8 depend only on existing docstrings/notebooks. D6 depends on this repo's results, not
on any integration work.

---

## 9. What stays in `transformer-ts-ranking`

| Stays in the benchmark | Flows to the library docs as |
|---|---|
| Ranking computation (Friedman/Nemenyi/CD, leaderboards) | An **exported figure/table** embedded in `benchmark/results.md` |
| `runtime_evidence.json` (empirical model behavior) | Input to the model cards' `runtime:` block (integration RFC §12) |

The docs site **embeds** the benchmark's published outputs; it never recomputes them. A small
`make sync-benchmark-assets` (or a release step) copies the current leaderboard PNG/CSV + CD diagram
into `docs/benchmark/`.

---

## 10. Relationship to the integration layer — and why docs can go first

The two efforts share **one source**: a model's description. To avoid maintaining it twice:

- **If the integration model cards exist**, the docs' per-model card block renders them.
- **If not yet**, the docs use a short front-matter block now, and swap to the generated card later
  — a one-line include change, not a rebuild.

This is a *content-sharing* seam, not a build dependency. Therefore:

> **Documentation can be implemented before the integration layer.** It is the most decoupled work
> item in the whole plan: it touches no runtime code, no benchmark eligibility, no running
> experiments, and has none of integration's parity gates or "never mid-run" constraints. It also
> lives in the library's own repo, like the integration work, so it does not touch this benchmark's
> submodule.

Only **two** items have any ordering at all, and both are soft:
- **D6** (results section) wants a completed benchmark run — nice to have, not required to start.
- **Per-model card block** is nicer once integration §6 exists — but has a working fallback.

Recommended: **start docs now**, in parallel with the running experiments, in the library repo.
It delivers community value immediately and de-risks the rest.

---

## 11. Open questions / risks

- **Notebook execution in CI** — model demos may need a GPU or be slow. Default to render-as-saved
  for model demos; execute only the light tutorial. Revisit if freshness becomes an issue.
- **Docstring coverage gate** — introduce `interrogate` at the *current* coverage level first, then
  ratchet up, so the gate never blocks unrelated PRs on day one.
- **Model-card double-source** — mitigated by the include/fallback in §7/§10; enforce "card text
  lives in one file only" in review.
- **Docs/code drift** — largely prevented by autodoc; the residual risk is hand-written guides going
  stale, mitigated by doctest-able snippets where practical.
- **Math rendering** — confirm `arithmatex` + MathJax config early on one equation-heavy page.

---

## 12. Appendix — concrete sketches

**`mkdocs.yml` (core)**

```yaml
site_name: s-transformers-lib
theme:
  name: material
  features: [navigation.sections, navigation.top, search.suggest, content.code.copy]
  palette:
    - scheme: default
      toggle: { icon: material/weather-night, name: Dark }
    - scheme: slate
      toggle: { icon: material/weather-sunny, name: Light }
plugins:
  - search
  - gen-files: { scripts: [scripts/gen_ref_pages.py] }
  - literate-nav: { nav_file: SUMMARY.md }
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true
            members_order: source
  - mkdocs-jupyter: { execute: false }
markdown_extensions:
  - pymdownx.arithmatex: { generic: true }
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
extra_javascript:
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
nav:
  - Home: index.md
  - Tutorials: [tutorials/first-forecast.md, tutorials/train-and-evaluate.md]
  - How-to: [how-to/add-a-model.md, how-to/normalization.md, how-to/dataloaders.md, how-to/choose-a-model.md]
  - Models: models/            # literate-nav expands
  - Explanation: [explanation/unified-contract.md, explanation/model-families.md]
  - API Reference: reference/   # gen-files + literate-nav
  - Benchmark: benchmark/results.md
```

**Per-model page (`docs/models/patchtst.md`)**

```markdown
---
title: PatchTST
---
{% include "cards/patchtst.md" %}   <!-- model card block: one source -->

## API
::: s_transformers_lib.models.patchtst

## Demo
[Open the notebook](../../examples/patchtst_demo.ipynb)
```

**`pyproject.toml` docs extra**

```toml
[project.optional-dependencies]
docs = [
  "mkdocs-material", "mkdocstrings[python]", "mkdocs-gen-files",
  "mkdocs-literate-nav", "mkdocs-jupyter", "mike",
]
```

**Read the Docs (`.readthedocs.yaml`)**

```yaml
version: 2
build: { os: ubuntu-24.04, tools: { python: "3.11" } }
mkdocs: { configuration: mkdocs.yml }
python: { install: [{ method: pip, path: ".", extra_requirements: ["docs"] }] }
```
