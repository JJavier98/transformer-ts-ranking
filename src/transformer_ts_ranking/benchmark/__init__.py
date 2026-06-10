"""Benchmark execution engine for the transformer-ts-ranking pipeline.

This package owns everything from window-dataset construction through
per-epoch training, result persistence, and full-run orchestration.
"""

from .engine import BenchmarkEngine, RunResult
from .runner import BenchmarkRunner, BenchmarkRunnerConfig
from .wandb_logger import WandbConfig, WandbLogger

__all__ = [
    "BenchmarkEngine",
    "RunResult",
    "BenchmarkRunner",
    "BenchmarkRunnerConfig",
    "WandbConfig",
    "WandbLogger",
]
