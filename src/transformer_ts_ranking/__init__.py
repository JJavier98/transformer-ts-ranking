"""Utilities for benchmarking and ranking S-TransformerTS models.

The package is organized as a pipeline of small layers so the benchmark can be
bootstrapped and validated incrementally:

- discovery: inventory, contract checks and runtime compatibility review
- data: dataset-specific loading and preprocessing
- adapters: family-level contracts derived from the capability matrix
- evaluation: smoke checks and metric computation
"""

__version__ = "0.1.0"
