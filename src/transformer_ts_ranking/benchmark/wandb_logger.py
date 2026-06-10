"""Weights & Biases experiment logger for the benchmark engine.

Wraps the wandb SDK so the engine and runner can log metrics without
importing wandb directly.  When W&B is disabled (``enabled=False`` or
``WANDB_MODE=disabled``) every method is a no-op, so no code path in
the engine needs to check ``if wandb_enabled``.

Usage::

    logger = WandbLogger.from_config(wandb_cfg, run_meta={...})
    logger.log_epoch(epoch=0, train_loss=0.5, val_loss=0.45, ...)
    logger.log_final(mae=0.3, mse=0.1, ...)
    logger.finish()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WandbConfig:
    """Hydra-compatible W&B configuration.

    Matches the fields in ``configs/hydra/wandb/*.yaml``.
    """

    enabled: bool = False
    project: str = "transformer-ts-ranking"
    entity: str | None = None
    mode: str = "online"          # online | offline | disabled
    log_model: bool = False
    tags: list[str] = field(default_factory=list)
    group_by: str = "model_dataset"

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> "WandbConfig":
        """Build a ``WandbConfig`` from a Hydra/OmegaConf DictConfig node.

        Args:
            cfg: OmegaConf node or plain dict with wandb config fields.

        Returns:
            Populated ``WandbConfig`` instance.
        """
        from omegaconf import OmegaConf  # noqa: PLC0415

        d = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "_metadata") else dict(cfg)
        return cls(
            enabled=bool(d.get("enabled", False)),
            project=str(d.get("project", "transformer-ts-ranking")),
            entity=d.get("entity") or None,
            mode=str(d.get("mode", "online")),
            log_model=bool(d.get("log_model", False)),
            tags=list(d.get("tags", [])),
            group_by=str(d.get("group_by", "model_dataset")),
        )


class WandbLogger:
    """Thin façade over the wandb SDK.

    One ``WandbLogger`` instance corresponds to one W&B *run* — that is,
    one (model, dataset, horizon, seed) combination.  The logger is created
    by the engine at the start of each trial and finished when the trial ends.

    When ``enabled=False`` every call is a silent no-op.
    """

    def __init__(
        self,
        config: WandbConfig,
        run_meta: dict[str, Any],
        env_file: Path | None = None,
    ) -> None:
        """Initialise and optionally start a W&B run.

        Args:
            config: W&B configuration.
            run_meta: Metadata dict logged as ``wandb.config`` for the run.
                Typical keys: model_name, dataset_name, horizon, seed,
                param_count, n_channels, seq_len.
            env_file: Optional path to a ``.env`` file containing
                ``WANDB_API_KEY`` etc.  Loaded via python-dotenv if present.
        """
        self._enabled = config.enabled and config.mode != "disabled"
        self._run: Any = None

        if not self._enabled:
            return

        self._load_env(env_file)

        try:
            import wandb  # noqa: PLC0415

            model_name = run_meta.get("model_name", "unknown")
            dataset_name = run_meta.get("dataset_name", "unknown")
            horizon = run_meta.get("horizon", 0)
            seed = run_meta.get("seed", 0)

            group = (
                f"{model_name}_{dataset_name}"
                if config.group_by == "model_dataset"
                else model_name
            )

            self._run = wandb.init(
                project=config.project,
                entity=config.entity,
                mode=config.mode,
                config=run_meta,
                name=f"{model_name}_{dataset_name}_h{horizon}_s{seed}",
                group=group,
                tags=list(config.tags),
                reinit=True,
            )
        except Exception as exc:
            # Never let W&B errors crash the benchmark.
            print(f"[wandb] Warning: could not initialise run — {exc}")
            self._enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        epoch_time_s: float | None = None,
        peak_gpu_mb: float | None = None,
    ) -> None:
        """Log per-epoch training metrics.

        Args:
            epoch: Zero-based epoch index.
            train_loss: Mean training loss for the epoch.
            val_loss: Mean validation loss (``None`` if no val set).
            epoch_time_s: Wall-clock seconds for the epoch.
            peak_gpu_mb: Peak GPU memory allocated during the epoch (MB).
        """
        if not self._enabled or self._run is None:
            return
        payload: dict[str, Any] = {"epoch": epoch, "train/loss": train_loss}
        if val_loss is not None and val_loss == val_loss:  # filter NaN
            payload["val/loss"] = val_loss
        if epoch_time_s is not None:
            payload["train/epoch_time_s"] = epoch_time_s
        if peak_gpu_mb is not None and peak_gpu_mb == peak_gpu_mb:
            payload["system/peak_gpu_mb"] = peak_gpu_mb
        try:
            self._run.log(payload, step=epoch)
        except Exception:
            pass

    def log_final(
        self,
        mae: float,
        mse: float,
        rmse: float,
        train_time_s: float | None = None,
        inference_latency_ms: float | None = None,
        peak_gpu_mb: float | None = None,
        smape: float | None = None,
        mase: float | None = None,
        owa: float | None = None,
    ) -> None:
        """Log final test-set metrics as a W&B summary.

        Args:
            mae: Mean Absolute Error on the test split (original scale).
            mse: Mean Squared Error.
            rmse: Root Mean Squared Error.
            train_time_s: Total training wall-clock time.
            inference_latency_ms: Per-sample inference latency.
            peak_gpu_mb: Peak GPU memory over the full run.
            smape: sMAPE (M4 track only).
            mase: MASE (M4 track only).
            owa: OWA (M4 track only).
        """
        if not self._enabled or self._run is None:
            return
        summary: dict[str, Any] = {
            "test/mae": mae,
            "test/mse": mse,
            "test/rmse": rmse,
        }
        if train_time_s is not None:
            summary["train/total_time_s"] = train_time_s
        if inference_latency_ms is not None:
            summary["test/inference_latency_ms"] = inference_latency_ms
        if peak_gpu_mb is not None and peak_gpu_mb == peak_gpu_mb:
            summary["system/peak_gpu_mb_total"] = peak_gpu_mb
        if smape is not None and smape == smape:
            summary["test/smape"] = smape
        if mase is not None and mase == mase:
            summary["test/mase"] = mase
        if owa is not None and owa == owa:
            summary["test/owa"] = owa
        try:
            self._run.summary.update(summary)
        except Exception:
            pass

    def finish(self, error: str | None = None) -> None:
        """Close the W&B run.

        Args:
            error: If the run failed, pass the error message so W&B marks
                the run as crashed rather than finished.
        """
        if not self._enabled or self._run is None:
            return
        try:
            if error:
                self._run.finish(exit_code=1)
            else:
                self._run.finish()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: WandbConfig,
        run_meta: dict[str, Any],
        repo_root: Path | None = None,
    ) -> "WandbLogger":
        """Create a logger, loading the ``.env`` file from the repo root.

        Args:
            config: W&B configuration.
            run_meta: Per-run metadata dict.
            repo_root: Repository root path used to locate ``.env``.

        Returns:
            Initialised ``WandbLogger`` instance (no-op if disabled).
        """
        env_file = (repo_root / ".env") if repo_root else None
        return cls(config=config, run_meta=run_meta, env_file=env_file)

    @classmethod
    def disabled(cls) -> "WandbLogger":
        """Return a no-op logger without initialising any W&B run."""
        return cls(config=WandbConfig(enabled=False), run_meta={})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_env(env_file: Path | None) -> None:
        """Load the .env file into os.environ via python-dotenv if available."""
        if env_file is None or not env_file.exists():
            return
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
            load_dotenv(env_file, override=False)
        except ImportError:
            # python-dotenv is optional; fall back to manual parse.
            with env_file.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
