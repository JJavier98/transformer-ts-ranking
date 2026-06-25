"""Hydra-decorated entry point for the full benchmark pipeline.

Launch examples::

    # Standard long-term run with W&B online logging
    conda run -n torch_env python scripts/launch_benchmark.py \\
        training=standard experiment=long_term wandb=online

    # Smoke run on CPU, no W&B
    conda run -n torch_env python scripts/launch_benchmark.py \\
        training=smoke experiment=long_term wandb=disabled

    # Full benchmark (both tracks), paper-ready settings
    conda run -n torch_env python scripts/launch_benchmark.py \\
        training=paper_ready experiment=full wandb=online

    # Override single model / dataset from CLI
    conda run -n torch_env python scripts/launch_benchmark.py \\
        training=smoke experiment=long_term wandb=disabled \\
        experiment.models=[itransformer,patchtst] experiment.datasets=[ETTh1]

    # Hydra multi-run sweep across horizons
    conda run -n torch_env python scripts/launch_benchmark.py --multirun \\
        training=standard experiment=long_term wandb=online \\
        experiment.horizons=[96],[192],[336],[720]

All Hydra outputs (logs, .hydra/ config snapshots) go to ``outputs/`` which is
gitignored.  Multi-run sweeps go to ``multirun/`` (also gitignored).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# ---- Make the package importable without installation --------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "s-transformers-lib"))

log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(_REPO_ROOT / "configs" / "hydra"),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Compose the Hydra config and launch a ``BenchmarkRunner`` run.

    Args:
        cfg: Hydra-composed DictConfig with keys ``training``, ``experiment``,
            and ``wandb``.
    """
    from transformer_ts_ranking.benchmark.runner import (  # noqa: PLC0415
        BenchmarkRunner,
        BenchmarkRunnerConfig,
    )
    from transformer_ts_ranking.benchmark.wandb_logger import (  # noqa: PLC0415
        WandbConfig,
    )

    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # ---- W&B config ---------------------------------------------------
    wandb_cfg = WandbConfig.from_omegaconf(cfg.wandb)

    # ---- Runner config ------------------------------------------------
    training = cfg.training
    experiment = cfg.experiment

    # Optional results_dir override — use for parallel runs on separate GPUs
    # to avoid write contention on the shared parquet file.
    # Pass as: +results_dir=results/raw/long_term
    results_dir = cfg.get("results_dir", None)
    results_dir_path = (_REPO_ROOT / results_dir) if results_dir else None

    runner_cfg = BenchmarkRunnerConfig(
        repo_root=_REPO_ROOT,
        results_dir=results_dir_path,
        # All training hyperparameters come from the training config group.
        device=str(training.get("device", "cpu")),
        epochs=int(training.get("epochs", 20)),
        batch_size=int(training.get("batch_size", 32)),
        patience=int(training.get("patience", 5)),
        lr=float(training.get("lr", 1e-4)),
        seeds=list(training.get("seeds", [42, 123, 2026])),
        dry_run=bool(training.get("dry_run", False)),
        m4_seq_len=int(training.get("m4_seq_len", 96)),
        # Scope (what to run) comes from the experiment config group.
        tasks=list(experiment.get("tasks", ["long_term", "m4"])),
        models=list(experiment.models) if experiment.get("models") else None,
        datasets=list(experiment.datasets) if experiment.get("datasets") else None,
        horizons=list(experiment.horizons) if experiment.get("horizons") else None,
        wandb=wandb_cfg,
    )

    log.info(
        "Starting benchmark: tasks=%s  device=%s  epochs=%d  seeds=%s",
        runner_cfg.tasks,
        runner_cfg.device,
        runner_cfg.epochs,
        runner_cfg.seeds,
    )

    runner = BenchmarkRunner(runner_cfg)
    results = runner.run()

    log.info("Benchmark finished.  %d runs recorded.", len(results))
    if not results.empty:
        log.info(
            "Summary:\n%s",
            results[["model_name", "dataset_name", "horizon", "seed", "mae", "mse"]]
            .to_string(index=False),
        )


if __name__ == "__main__":
    main()
