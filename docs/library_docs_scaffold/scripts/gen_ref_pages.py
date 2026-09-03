"""Auto-generate the API Reference page tree from the package source.

Standard mkdocstrings recipe (mkdocs-gen-files + mkdocs-literate-nav): walk the
library source, emit one Markdown stub per module containing a single
``::: <dotted.path>`` autodoc directive, and write a ``SUMMARY.md`` so
mkdocs-literate-nav builds the navigation automatically. New modules/models
therefore appear in the docs with **no manual nav edits**.

Runs at ``mkdocs build`` time; writes only to the in-memory virtual docs tree
(via ``mkdocs_gen_files.open``), never to disk under ``docs/``.

Adjust ``SRC_DIR`` / ``ROOT_PACKAGE`` if the library's src-layout differs.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

# The library uses a src-layout where ``src/`` maps to the importable package
# ``s_transformers_lib`` (see docs/documentation_site_design.md §6). If the
# layout is ``src/s_transformers_lib/``, point SRC_DIR at that inner directory.
SRC_DIR = "src"
ROOT_PACKAGE = "s_transformers_lib"

nav = mkdocs_gen_files.Nav()
src = Path(__file__).resolve().parent.parent / SRC_DIR

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts and parts[-1] == "__main__":
        continue
    if not parts:
        continue

    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join((ROOT_PACKAGE, *parts))
        fd.write(f"# `{identifier}`\n\n::: {identifier}\n")
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src.parent))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
