"""Auto-generate one documentation page per registered model.

For every name returned by ``s_transformers_lib.list_models()`` this emits
``models/<name>.md`` composed of three blocks, each from a **single source**
(docs/documentation_site_design.md §7):

1. **Card** — inlined from ``docs/cards/<name>.md`` when it exists (the file the
   integration layer's model-card generator will produce). Until then a short
   stub is written, so the page works today and gains the real card later with
   no page change (see the integration RFC §10 fallback).
2. **API** — ``::: s_transformers_lib.models.<name>`` (mkdocstrings autodoc).
3. **Demo** — a link to ``examples/<name>_demo.ipynb``.

Also writes ``models/SUMMARY.md`` for mkdocs-literate-nav. Because it is driven
by ``list_models()``, a newly registered model appears here automatically.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

ROOT_PACKAGE = "s_transformers_lib"

# Resolve the registered models. If the package is not importable at build time
# (e.g. a docs-only environment), fail loudly — mkdocstrings needs it installed
# anyway to render the API blocks.
try:
    from s_transformers_lib import list_models  # type: ignore
    model_names = sorted(list_models())
except Exception as exc:  # pragma: no cover - build-time diagnostic
    raise RuntimeError(
        "Could not import list_models() to generate model pages. Install the "
        "library into the docs environment (pip install .[docs])."
    ) from exc

# Real card text lives in exactly one place: docs/cards/<name>.md.
CARDS_DIR = Path(__file__).resolve().parent.parent / "docs" / "cards"

nav = mkdocs_gen_files.Nav()

for name in model_names:
    page = Path("models", f"{name}.md")
    nav[(name,)] = f"{name}.md"

    card_file = CARDS_DIR / f"{name}.md"
    if card_file.is_file():
        card_block = card_file.read_text()
    else:
        card_block = (
            f"# {name}\n\n"
            "!!! note \"Model card pending\"\n"
            "    A generated model card will appear here once the integration "
            "layer's card generator is in place. See the API and demo below.\n"
        )

    with mkdocs_gen_files.open(page, "w") as fd:
        fd.write(card_block.rstrip() + "\n\n")
        fd.write("## API\n\n")
        fd.write(f"::: {ROOT_PACKAGE}.models.{name}\n\n")
        fd.write("## Demo\n\n")
        fd.write(
            f"A runnable demo notebook lives at `examples/{name}_demo.ipynb` in "
            "the repository.\n"
        )

with mkdocs_gen_files.open("models/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
