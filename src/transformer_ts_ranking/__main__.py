"""CLI entry-point for the benchmark package.

This module intentionally stays minimal so ``python -m transformer_ts_ranking``
delegates all real command wiring to ``transformer_ts_ranking.cli``.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
