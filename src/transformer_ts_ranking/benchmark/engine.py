"""Single-run benchmark engine: one model × dataset/frequency × horizon × seed.

The engine is responsible for:
  1. Instantiating the model via the library factory.
  2. Training epoch by epoch (epoch=1 loop so per-epoch metrics can be logged).
  3. Collecting train/val loss, epoch wall time, and peak GPU memory per epoch.
  4. Running inference on the test set and inverse-scaling predictions.
  5. Computing MAE, MSE, and RMSE on the original scale.
  6. Returning a structured ``RunResult`` and a list of per-epoch log dicts.

Design invariants (see CLAUDE.md):
  - Temporal split before windowing: DataLoader windows are pre-split.
  - Inverse-scale before metrics: scaler.inverse_transform() is called on
    model predictions before any metric is computed.
  - No ``if model_name ==`` in the engine: model differences are handled by
    the library internally; the engine always passes the same canonical batch.

W&B integration:
  - Callers pass an optional ``WandbLogger`` to ``run_long_term()`` / ``run_m4()``.
  - A ``WandbLogger.disabled()`` instance is used when no logger is provided so
    every call is a no-op and the engine needs no ``if`` guards.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .wandb_logger import WandbLogger  # noqa: TID252

# The library submodule is added to sys.path by runner.py / the CLI shim.
# Import lazily so tests that mock the library can override it.


@dataclass
class RunResult:
    """Metrics and metadata produced by one benchmark run.

    One row in ``results/raw/results_raw.parquet`` corresponds to one
    ``RunResult`` (after flattening to a plain dict via ``to_record()``).
    """

    # Identity
    model_name: str
    dataset_name: str
    horizon: int
    seed: int
    task: str  # "long_term" or "m4"

    # Accuracy metrics (original scale)
    mae: float
    mse: float
    rmse: float

    # M4-specific metrics (NaN for long-term runs)
    smape: float = float("nan")
    mase: float = float("nan")
    owa: float = float("nan")

    # Efficiency
    train_time_s: float = float("nan")
    inference_latency_ms: float = float("nan")
    peak_gpu_mb: float = float("nan")
    param_count: int = 0

    # Training outcome
    best_val_loss: float = float("nan")
    epochs_trained: int = 0
    stopped_early: bool = False
    error: str | None = None

    # Per-epoch log entries (not persisted in parquet; go to JSONL separately)
    epoch_logs: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_record(self) -> dict[str, Any]:
        """Flatten to a plain dict for parquet/CSV serialisation.

        The ``epoch_logs`` field is excluded — those are written to a
        separate JSONL file.
        """
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "horizon": self.horizon,
            "seed": self.seed,
            "task": self.task,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "smape": self.smape,
            "mase": self.mase,
            "owa": self.owa,
            "train_time_s": self.train_time_s,
            "inference_latency_ms": self.inference_latency_ms,
            "peak_gpu_mb": self.peak_gpu_mb,
            "param_count": self.param_count,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
            "stopped_early": self.stopped_early,
            "error": self.error,
        }


def _count_parameters(model: Any) -> int:
    """Count trainable parameters of a PyTorch model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _peak_gpu_mb(device: str) -> float:
    """Return peak GPU memory allocated since the last reset, in megabytes."""
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _reset_peak_gpu(device: str) -> None:
    """Reset the peak GPU memory counter."""
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _strip_future_orig(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove non-model keys that DataLoader collated but models don't accept."""
    excluded = {"future_orig", "train_orig", "test_orig", "mean_enc", "std_enc", "_series_id"}
    return {k: v for k, v in batch.items() if k not in excluded}


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((pred - target) ** 2))


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(_mse(pred, target)))


class BenchmarkEngine:
    """Runs a single model × dataset/horizon × seed benchmark trial.

    Usage::

        engine = BenchmarkEngine(device="cuda", epochs=20, batch_size=32,
                                 patience=5, repo_root=Path(...))
        result = engine.run_long_term(
            model_name="itransformer",
            model_config={...},
            dataset=loaded_dataset,
            pred_len=96,
            seed=42,
        )
    """

    def __init__(
        self,
        device: str = "cpu",
        epochs: int = 20,
        batch_size: int = 32,
        patience: int = 5,
        lr: float = 1e-4,
        num_workers: int = 0,
        repo_root: Path | None = None,
    ) -> None:
        """Initialise the engine with shared training hyperparameters.

        Args:
            device: Torch device string (``"cpu"`` or ``"cuda"``).
            epochs: Maximum training epochs.
            batch_size: DataLoader batch size.
            patience: Early-stopping patience (epochs without val improvement).
            lr: Learning rate passed to the default Adam optimizer.
            num_workers: DataLoader worker processes.
            repo_root: Benchmark repository root (used to locate submodule).
        """
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.num_workers = num_workers
        self.repo_root = repo_root or Path(__file__).resolve().parents[4]
        self._ensure_library_on_path()

    def _ensure_library_on_path(self) -> None:
        """Add the s-transformers-lib root to sys.path.

        The library's registry uses ``src.models.<key>`` as the module path,
        so the library *root* (not its ``src/`` subdirectory) must be on
        ``sys.path`` so that ``import src.models.itransformer`` resolves
        correctly.
        """
        lib_root = str(self.repo_root / "s-transformers-lib")
        if lib_root not in sys.path:
            sys.path.insert(0, lib_root)

    # ------------------------------------------------------------------
    # Long-term benchmark
    # ------------------------------------------------------------------

    def run_long_term(
        self,
        model_name: str,
        model_config: dict[str, Any],
        dataset: Any,
        pred_len: int,
        seed: int,
        wandb_logger: WandbLogger | None = None,
    ) -> RunResult:
        """Train and evaluate one model on one long-term dataset/horizon/seed.

        Args:
            model_name: Canonical model key (e.g. ``"itransformer"``).
            model_config: Config dict for ``create_model()``.
            dataset: ``LoadedLongTermDataset`` instance.
            pred_len: Forecast horizon.
            seed: Random seed (applied to Torch and NumPy).
            wandb_logger: Optional W&B logger for this run.  When ``None`` a
                no-op logger is used so the engine needs no ``if`` guards.

        Returns:
            Fully populated ``RunResult``.
        """
        from ..benchmark.window_dataset import LongTermWindowDataset

        logger = wandb_logger or WandbLogger.disabled()
        self._set_seed(seed)

        train_ds = LongTermWindowDataset(dataset, pred_len, "train")
        val_ds = LongTermWindowDataset(dataset, pred_len, "val")
        test_ds = LongTermWindowDataset(dataset, pred_len, "test")

        if len(train_ds) == 0 or len(test_ds) == 0:
            err = "Insufficient windows for this horizon/split combination."
            logger.finish(error=err)
            return RunResult(
                model_name=model_name,
                dataset_name=dataset.dataset_name,
                horizon=pred_len,
                seed=seed,
                task="long_term",
                mae=float("nan"),
                mse=float("nan"),
                rmse=float("nan"),
                error=err,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        ) if len(val_ds) > 0 else None

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

        try:
            result = self._train_and_eval_long_term(
                model_name=model_name,
                model_config=model_config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                scaler=dataset.scaler,
                n_channels=len(dataset.feature_columns),
                pred_len=pred_len,
                seed=seed,
                dataset_name=dataset.dataset_name,
                logger=logger,
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.finish(error=err_msg)
            result = RunResult(
                model_name=model_name,
                dataset_name=dataset.dataset_name,
                horizon=pred_len,
                seed=seed,
                task="long_term",
                mae=float("nan"),
                mse=float("nan"),
                rmse=float("nan"),
                error=err_msg,
            )
        return result

    def _train_and_eval_long_term(
        self,
        model_name: str,
        model_config: dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader,
        scaler: Any,
        n_channels: int,
        pred_len: int,
        seed: int,
        dataset_name: str,
        logger: WandbLogger,
    ) -> RunResult:
        """Execute the full training loop and test evaluation."""
        from src.interfaces.forecasting import TrainingConfig  # noqa: PLC0415
        from src.models import create_model                    # noqa: PLC0415

        model = create_model(model_name, config=model_config)
        model.to(self.device)
        param_count = _count_parameters(model)

        training_cfg = TrainingConfig(epochs=1, device=self.device, lr=self.lr)
        optimizer = model._build_default_optimizer(training_cfg)

        best_val_loss = float("inf")
        patience_counter = 0
        epoch_logs: list[dict[str, Any]] = []
        total_train_time = 0.0
        stopped_early = False

        _reset_peak_gpu(self.device)
        run_id = f"{model_name}_{dataset_name}_h{pred_len}_s{seed}"

        for epoch in range(self.epochs):
            t_epoch_start = time.perf_counter()

            # One epoch of training via the library's fit() API.
            # Passing the optimizer externally preserves optimiser state
            # across the epoch loop.
            model.fit(
                self._filtered_loader(train_loader),
                self._filtered_loader(val_loader) if val_loader else None,
                training=training_cfg,
                optimizer=optimizer,
            )

            epoch_time = time.perf_counter() - t_epoch_start
            total_train_time += epoch_time
            gpu_mb = _peak_gpu_mb(self.device)

            train_loss = model.history_["train_loss"][-1] if model.history_["train_loss"] else float("nan")
            val_loss_val = model.history_["val_loss"][-1] if model.history_["val_loss"] else float("nan")

            epoch_log: dict[str, Any] = {
                "run_id": run_id,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "horizon": pred_len,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_val,
                "epoch_time_s": epoch_time,
                "peak_gpu_mb": gpu_mb,
            }
            epoch_logs.append(epoch_log)

            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss_val if not np.isnan(val_loss_val) else None,
                epoch_time_s=epoch_time,
                peak_gpu_mb=gpu_mb if not np.isnan(gpu_mb) else None,
            )

            # Early stopping on validation loss.
            if not np.isnan(val_loss_val):
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        stopped_early = True
                        break

        # ---- Inference on test set -------------------------------------------
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        latencies: list[float] = []

        model.eval()
        model.to(self.device)

        with torch.no_grad():
            for batch in test_loader:
                future_orig = batch.pop("future_orig").numpy()  # (B, pred_len, C)
                clean_batch = _strip_future_orig(batch)
                x = clean_batch["x"].to(self.device).float()
                x_mark = clean_batch.get("x_mark")
                y_mark = clean_batch.get("y_mark")
                if x_mark is not None:
                    x_mark = x_mark.to(self.device).float()
                if y_mark is not None:
                    y_mark = y_mark.to(self.device).float()

                from src.interfaces.forecasting import ForecastInput  # noqa: PLC0415

                forecast_input = ForecastInput(x=x, x_mark=x_mark, y_mark=y_mark)

                t0 = time.perf_counter()
                output = model.predict(forecast_input, device=self.device)
                latencies.append((time.perf_counter() - t0) * 1000 / x.shape[0])

                # predictions: (B, pred_len, C) — still in scaled space
                preds_scaled = output.prediction.cpu().numpy()

                # Inverse-scale each sample in the batch independently using
                # the scaler that was fitted on the training split only.
                preds_orig = scaler.inverse_transform(
                    preds_scaled.reshape(-1, n_channels)
                ).reshape(preds_scaled.shape)

                all_preds.append(preds_orig)
                all_targets.append(future_orig)

        preds_np = np.concatenate(all_preds, axis=0)   # (N, pred_len, C)
        targets_np = np.concatenate(all_targets, axis=0)

        mae_val = _mae(preds_np, targets_np)
        mse_val = _mse(preds_np, targets_np)
        rmse_val = _rmse(preds_np, targets_np)
        avg_latency = float(np.mean(latencies)) if latencies else float("nan")
        final_gpu_mb = _peak_gpu_mb(self.device)

        logger.log_final(
            mae=mae_val,
            mse=mse_val,
            rmse=rmse_val,
            train_time_s=total_train_time,
            inference_latency_ms=avg_latency,
            peak_gpu_mb=final_gpu_mb if not np.isnan(final_gpu_mb) else None,
        )
        logger.finish()

        return RunResult(
            model_name=model_name,
            dataset_name=dataset_name,
            horizon=pred_len,
            seed=seed,
            task="long_term",
            mae=mae_val,
            mse=mse_val,
            rmse=rmse_val,
            train_time_s=total_train_time,
            inference_latency_ms=avg_latency,
            peak_gpu_mb=final_gpu_mb,
            param_count=param_count,
            best_val_loss=best_val_loss if best_val_loss != float("inf") else float("nan"),
            epochs_trained=len(epoch_logs),
            stopped_early=stopped_early,
            epoch_logs=epoch_logs,
        )

    # ------------------------------------------------------------------
    # M4 short-term benchmark
    # ------------------------------------------------------------------

    def run_m4(
        self,
        model_name: str,
        model_config: dict[str, Any],
        dataset: Any,
        seq_len: int,
        seed: int,
        wandb_logger: WandbLogger | None = None,
    ) -> RunResult:
        """Train and evaluate one model on one M4 frequency slice.

        Args:
            model_name: Canonical model key.
            model_config: Config dict for ``create_model()``.
            dataset: ``LoadedM4Dataset`` instance.
            seq_len: Encoder input length.
            seed: Random seed.
            wandb_logger: Optional W&B logger for this run.

        Returns:
            Fully populated ``RunResult`` including sMAPE, MASE, OWA.
        """
        from ..benchmark.window_dataset import M4SeriesDataset

        logger = wandb_logger or WandbLogger.disabled()
        self._set_seed(seed)

        m4_ds = M4SeriesDataset(dataset, seq_len=seq_len)
        m4_loader = DataLoader(
            m4_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=_m4_collate_fn,
        )

        try:
            result = self._train_and_eval_m4(
                model_name=model_name,
                model_config=model_config,
                dataset=dataset,
                m4_loader=m4_loader,
                seq_len=seq_len,
                seed=seed,
                logger=logger,
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.finish(error=err_msg)
            result = RunResult(
                model_name=model_name,
                dataset_name=dataset.frequency_label,
                horizon=dataset.horizon,
                seed=seed,
                task="m4",
                mae=float("nan"),
                mse=float("nan"),
                rmse=float("nan"),
                error=err_msg,
            )
        return result

    def _train_and_eval_m4(
        self,
        model_name: str,
        model_config: dict[str, Any],
        dataset: Any,
        m4_loader: DataLoader,
        seq_len: int,
        seed: int,
        logger: WandbLogger,
    ) -> RunResult:
        """Full M4 training + OWA evaluation."""
        from src.interfaces.forecasting import ForecastInput, TrainingConfig  # noqa: PLC0415
        from src.models import create_model                                    # noqa: PLC0415
        from ..evaluation.m4_metrics import (                                  # noqa: PLC0415
            evaluate_m4_dataset,
            smape as _smape,
            mase as _mase,
        )

        model = create_model(model_name, config=model_config)
        model.to(self.device)
        param_count = _count_parameters(model)

        training_cfg = TrainingConfig(epochs=1, device=self.device, lr=self.lr)
        optimizer = model._build_default_optimizer(training_cfg)

        epoch_logs: list[dict[str, Any]] = []
        total_train_time = 0.0
        best_val_loss = float("inf")
        stopped_early = False
        run_id = f"{model_name}_{dataset.frequency_label}_s{seed}"

        _reset_peak_gpu(self.device)

        for epoch in range(self.epochs):
            t0 = time.perf_counter()
            model.fit(
                self._filtered_m4_loader(m4_loader),
                training=training_cfg,
                optimizer=optimizer,
            )
            epoch_time = time.perf_counter() - t0
            total_train_time += epoch_time
            train_loss = model.history_["train_loss"][-1] if model.history_["train_loss"] else float("nan")

            gpu_mb_m4 = _peak_gpu_mb(self.device)
            epoch_logs.append({
                "run_id": run_id,
                "model_name": model_name,
                "dataset_name": dataset.frequency_label,
                "horizon": dataset.horizon,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float("nan"),
                "epoch_time_s": epoch_time,
                "peak_gpu_mb": gpu_mb_m4,
            })
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                epoch_time_s=epoch_time,
                peak_gpu_mb=gpu_mb_m4 if not np.isnan(gpu_mb_m4) else None,
            )

        # ---- M4 inference: predict per series --------------------------------
        model.eval()
        predictions: dict[str, np.ndarray] = {}
        latencies: list[float] = []

        with torch.no_grad():
            for batch in DataLoader(
                M4SeriesDataset(dataset, seq_len=seq_len),  # type: ignore[arg-type]
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=_m4_collate_fn,
            ):
                x = batch["x"].to(self.device).float()
                x_mark = batch["x_mark"].to(self.device).float()
                series_ids = batch["_series_id"]  # list of strings

                forecast_input = ForecastInput(x=x, x_mark=x_mark)
                t0 = time.perf_counter()
                output = model.predict(forecast_input, device=self.device)
                latencies.append((time.perf_counter() - t0) * 1000 / x.shape[0])

                preds_scaled = output.prediction.cpu().numpy()  # (B, horizon, 1)

                for b_idx, sid in enumerate(series_ids):
                    series = dataset.series[sid]
                    mean_enc = float(np.mean(series.train_values))
                    std_enc = float(np.std(series.train_values) + 1e-8)
                    pred_orig = preds_scaled[b_idx, :, 0] * std_enc + mean_enc
                    predictions[sid] = pred_orig.astype(np.float64)

        eval_result = evaluate_m4_dataset(dataset=dataset, predictions=predictions)

        preds_all = np.array([predictions[sid] for sid in dataset.series_ids])
        targets_all = np.array([dataset.series[sid].test_values for sid in dataset.series_ids])
        mae_val = _mae(preds_all, targets_all)
        mse_val = _mse(preds_all, targets_all)
        rmse_val = _rmse(preds_all, targets_all)
        avg_latency_m4 = float(np.mean(latencies)) if latencies else float("nan")
        final_gpu_mb_m4 = _peak_gpu_mb(self.device)

        logger.log_final(
            mae=mae_val,
            mse=mse_val,
            rmse=rmse_val,
            train_time_s=total_train_time,
            inference_latency_ms=avg_latency_m4,
            peak_gpu_mb=final_gpu_mb_m4 if not np.isnan(final_gpu_mb_m4) else None,
            smape=eval_result.mean_smape,
            mase=eval_result.mean_mase,
            owa=eval_result.mean_owa,
        )
        logger.finish()

        return RunResult(
            model_name=model_name,
            dataset_name=dataset.frequency_label,
            horizon=dataset.horizon,
            seed=seed,
            task="m4",
            mae=mae_val,
            mse=mse_val,
            rmse=rmse_val,
            smape=eval_result.mean_smape,
            mase=eval_result.mean_mase,
            owa=eval_result.mean_owa,
            train_time_s=total_train_time,
            inference_latency_ms=avg_latency_m4,
            peak_gpu_mb=final_gpu_mb_m4,
            param_count=param_count,
            best_val_loss=float("nan"),
            epochs_trained=len(epoch_logs),
            stopped_early=stopped_early,
            epoch_logs=epoch_logs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set all relevant random seeds for reproducibility."""
        import random  # noqa: PLC0415
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _filtered_loader(loader: DataLoader | None) -> Any:
        """Return an iterable that strips non-model keys from each batch."""
        if loader is None:
            return None
        return _FilteredLoader(loader)

    @staticmethod
    def _filtered_m4_loader(loader: DataLoader) -> Any:
        """Return an iterable that strips M4 metadata keys from each batch."""
        return _FilteredM4Loader(loader)


class _FilteredLoader:
    """Wraps a DataLoader to strip benchmark-internal keys from each batch."""

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):  # type: ignore[override]
        for batch in self._loader:
            yield _strip_future_orig(batch)

    def __len__(self) -> int:
        return len(self._loader)


class _FilteredM4Loader:
    """Wraps a DataLoader to strip M4 metadata keys from each batch."""

    _M4_META = {"train_orig", "test_orig", "mean_enc", "std_enc", "_series_id"}

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):  # type: ignore[override]
        for batch in self._loader:
            yield {k: v for k, v in batch.items() if k not in self._M4_META}

    def __len__(self) -> int:
        return len(self._loader)


def _m4_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate for M4: stack tensors, keep metadata as lists."""
    tensor_keys = {"x", "y", "x_mark", "y_mark"}
    result: dict[str, Any] = {}
    for key in tensor_keys:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    for key in ("train_orig", "test_orig"):
        if key in batch[0]:
            result[key] = [item[key] for item in batch]
    for key in ("mean_enc", "std_enc"):
        if key in batch[0]:
            result[key] = [item[key] for item in batch]
    if "_series_id" in batch[0]:
        result["_series_id"] = [item["_series_id"] for item in batch]
    return result
