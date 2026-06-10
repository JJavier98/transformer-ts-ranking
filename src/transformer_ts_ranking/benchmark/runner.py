"""Full benchmark orchestrator: all models × datasets × horizons × seeds.

The runner reads the benchmark manifests, resolves eligible models, loads
datasets on demand, and dispatches each (model, dataset, horizon, seed)
combination to the ``BenchmarkEngine``.  Results are streamed to disk after
every run so a partial crash does not lose previously computed results.

Result layout:
    results/raw/results_raw.parquet   — one row per run
    results/raw/epoch_logs.jsonl      — one JSON line per (run, epoch)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..configuration import load_yaml
from .engine import BenchmarkEngine, RunResult
from .model_configs import build_long_term_config, build_m4_config, filter_config_for_model
from .wandb_logger import WandbConfig, WandbLogger


@dataclass
class BenchmarkRunnerConfig:
    """Configuration for a full benchmark run.

    Attributes:
        repo_root: Root of the benchmark repository.
        config_dir: Directory containing versioned YAML manifests.
        results_dir: Directory for parquet + JSONL result files.
        device: Torch device (``"cuda"`` or ``"cpu"``).
        epochs: Maximum training epochs per run.
        batch_size: Training DataLoader batch size.
        patience: Early-stopping patience in epochs.
        lr: Adam learning rate.
        seeds: Random seeds; one full run per seed.
        models: Restrict to these model names (``None`` → all eligible).
        datasets: Restrict to these dataset names (``None`` → all configured).
        horizons: Restrict to these horizons (``None`` → all configured).
        tasks: Which benchmark tracks to run (``"long_term"`` and/or ``"m4"``).
        dry_run: If ``True``, skip actual training (used for pipeline checks).
        m4_seq_len: Encoder input length used for all M4 series.
    """

    repo_root: Path
    config_dir: Path | None = None
    results_dir: Path | None = None
    device: str = "cpu"
    epochs: int = 20
    batch_size: int = 32
    patience: int = 5
    lr: float = 1e-4
    seeds: list[int] = field(default_factory=lambda: [42, 123, 2026])
    models: list[str] | None = None
    datasets: list[str] | None = None
    horizons: list[int] | None = None
    tasks: list[str] = field(default_factory=lambda: ["long_term", "m4"])
    dry_run: bool = False
    m4_seq_len: int = 96
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        if self.config_dir is None:
            self.config_dir = self.repo_root / "configs" / "benchmark"
        if self.results_dir is None:
            self.results_dir = self.repo_root / "results" / "raw"


class BenchmarkRunner:
    """Orchestrates the full benchmark run from manifests to parquet results.

    Usage::

        cfg = BenchmarkRunnerConfig(repo_root=Path("."), device="cuda", epochs=20)
        runner = BenchmarkRunner(cfg)
        runner.run()
    """

    def __init__(self, config: BenchmarkRunnerConfig) -> None:
        """Initialise the runner.

        Args:
            config: Full benchmark configuration.
        """
        self.cfg = config
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._parquet_path = self.results_dir / "results_raw.parquet"
        self._jsonl_path = self.results_dir / "epoch_logs.jsonl"

        self._engine = BenchmarkEngine(
            device=config.device,
            epochs=config.epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            lr=config.lr,
            repo_root=config.repo_root,
        )
        self._ensure_library_on_path()

    def _ensure_library_on_path(self) -> None:
        """Add submodule root to sys.path (registry uses src.models.* paths)."""
        lib_root = str(self.cfg.repo_root / "s-transformers-lib")
        if lib_root not in sys.path:
            sys.path.insert(0, lib_root)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full benchmark and return the accumulated results frame.

        Returns:
            DataFrame of all ``RunResult`` records written during this run.
        """
        all_records: list[dict[str, Any]] = []

        if "long_term" in self.cfg.tasks:
            records = self._run_long_term_track()
            all_records.extend(records)

        if "m4" in self.cfg.tasks:
            records = self._run_m4_track()
            all_records.extend(records)

        if all_records:
            frame = pd.DataFrame(all_records)
            # Merge with any previously persisted results.
            if self._parquet_path.exists():
                existing = pd.read_parquet(self._parquet_path)
                frame = pd.concat([existing, frame], ignore_index=True)
            frame.to_parquet(self._parquet_path, index=False)
            return frame

        if self._parquet_path.exists():
            return pd.read_parquet(self._parquet_path)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Long-term track
    # ------------------------------------------------------------------

    def _run_long_term_track(self) -> list[dict[str, Any]]:
        """Iterate all (model, dataset, horizon, seed) combinations."""
        from ..data.long_term import load_long_term_dataset

        manifest_paths = self._manifest_paths()
        capability = load_yaml(manifest_paths["capability_matrix"])
        eligible_models = self._eligible_models(capability, task="long_term")
        datasets_manifest = load_yaml(manifest_paths["long_term_datasets"])

        dataset_names = list(datasets_manifest.get("datasets", {}).keys())
        if self.cfg.datasets:
            dataset_names = [d for d in dataset_names if d in self.cfg.datasets]

        records: list[dict[str, Any]] = []

        for dataset_name in dataset_names:
            print(f"[long_term] Loading dataset: {dataset_name}")
            try:
                dataset = load_long_term_dataset(
                    repo_root=self.cfg.repo_root,
                    dataset_name=dataset_name,
                    manifest_path=manifest_paths["long_term_datasets"],
                )
            except Exception as exc:
                print(f"  ERROR loading {dataset_name}: {exc}")
                continue

            horizons = dataset.horizons
            if self.cfg.horizons:
                horizons = [h for h in horizons if h in self.cfg.horizons]

            for horizon in horizons:
                for model_name in eligible_models:
                    raw_cfg = build_long_term_config(
                        model_name=model_name,
                        n_channels=len(dataset.feature_columns),
                        pred_len=horizon,
                        seq_len=dataset.seq_len,
                        label_len=dataset.label_len,
                        freq=dataset.frequency,
                    )
                    model_cfg = filter_config_for_model(model_name, raw_cfg)
                    for seed in self.cfg.seeds:
                        tag = f"  [{model_name}] {dataset_name} h={horizon} seed={seed}"
                        if self.cfg.dry_run:
                            print(f"{tag} [DRY RUN — skipped]")
                            continue
                        print(tag)
                        run_meta = {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "horizon": horizon,
                            "seed": seed,
                            "task": "long_term",
                            "seq_len": dataset.seq_len,
                            "n_channels": len(dataset.feature_columns),
                        }
                        logger = WandbLogger.from_config(
                            self.cfg.wandb,
                            run_meta=run_meta,
                            repo_root=self.cfg.repo_root,
                        )
                        result = self._engine.run_long_term(
                            model_name=model_name,
                            model_config=model_cfg,
                            dataset=dataset,
                            pred_len=horizon,
                            seed=seed,
                            wandb_logger=logger,
                        )
                        self._persist_result(result)
                        records.append(result.to_record())
                        status = f"MAE={result.mae:.4f}" if result.error is None else f"ERROR: {result.error}"
                        print(f"    → {status}")

        return records

    # ------------------------------------------------------------------
    # M4 track
    # ------------------------------------------------------------------

    def _run_m4_track(self) -> list[dict[str, Any]]:
        """Iterate all (model, frequency, seed) combinations for M4."""
        from ..data.m4 import load_m4_dataset

        manifest_paths = self._manifest_paths()
        capability = load_yaml(manifest_paths["capability_matrix"])
        eligible_models = self._eligible_models(capability, task="m4")
        m4_manifest = load_yaml(manifest_paths["m4_datasets"])

        frequency_labels = list(m4_manifest.get("frequencies", {}).keys())
        if self.cfg.datasets:
            frequency_labels = [f for f in frequency_labels if f in self.cfg.datasets]

        records: list[dict[str, Any]] = []

        for freq_label in frequency_labels:
            print(f"[m4] Loading frequency: {freq_label}")
            try:
                dataset = load_m4_dataset(
                    repo_root=self.cfg.repo_root,
                    frequency_label=freq_label,
                    manifest_path=manifest_paths["m4_datasets"],
                )
            except Exception as exc:
                print(f"  ERROR loading M4/{freq_label}: {exc}")
                continue

            for model_name in eligible_models:
                raw_cfg = build_m4_config(
                    model_name=model_name,
                    seq_len=self.cfg.m4_seq_len,
                    horizon=dataset.horizon,
                )
                model_cfg = filter_config_for_model(model_name, raw_cfg)
                for seed in self.cfg.seeds:
                    tag = f"  [{model_name}] M4/{freq_label} seed={seed}"
                    if self.cfg.dry_run:
                        print(f"{tag} [DRY RUN — skipped]")
                        continue
                    print(tag)
                    run_meta = {
                        "model_name": model_name,
                        "dataset_name": freq_label,
                        "horizon": dataset.horizon,
                        "seed": seed,
                        "task": "m4",
                        "seq_len": self.cfg.m4_seq_len,
                    }
                    logger = WandbLogger.from_config(
                        self.cfg.wandb,
                        run_meta=run_meta,
                        repo_root=self.cfg.repo_root,
                    )
                    result = self._engine.run_m4(
                        model_name=model_name,
                        model_config=model_cfg,
                        dataset=dataset,
                        seq_len=self.cfg.m4_seq_len,
                        seed=seed,
                        wandb_logger=logger,
                    )
                    self._persist_result(result)
                    records.append(result.to_record())
                    status = f"OWA={result.owa:.4f}" if result.error is None else f"ERROR: {result.error}"
                    print(f"    → {status}")

        return records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _manifest_paths(self) -> dict[str, Path]:
        """Return the four benchmark manifest paths (materialise if missing)."""
        config_dir = self.cfg.config_dir
        paths = {
            "capability_matrix": config_dir / "model_capability_matrix.yaml",
            "long_term_datasets": config_dir / "long_term_datasets.yaml",
            "m4_datasets": config_dir / "m4_datasets.yaml",
            "training_presets": config_dir / "training_presets.yaml",
        }
        if not all(p.exists() for p in paths.values()):
            from ..bootstrap import materialize_bootstrap_manifests
            materialize_bootstrap_manifests(
                repo_root=self.cfg.repo_root,
                config_dir=config_dir,
                audit_output_dir=self.cfg.repo_root / "artifacts" / "audit",
            )
        return paths

    def _eligible_models(
        self,
        capability: dict[str, Any],
        task: str,
    ) -> list[str]:
        """Resolve eligible model names for a task from the capability matrix."""
        task_flag = "eligible_long_term" if task == "long_term" else "eligible_m4"
        eligible = [
            entry["model_name"]
            for entry in capability.get("models", [])
            if bool(entry.get(task_flag, False))
        ]
        if self.cfg.models:
            eligible = [m for m in eligible if m in self.cfg.models]
        return eligible

    def _persist_result(self, result: RunResult) -> None:
        """Append one run record to the JSONL epoch log and flush."""
        with self._jsonl_path.open("a") as fh:
            for log_entry in result.epoch_logs:
                fh.write(json.dumps(log_entry) + "\n")
