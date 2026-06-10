"""Thin CLI shim: adds src/ to sys.path then delegates to the benchmark CLI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from transformer_ts_ranking.cli import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
