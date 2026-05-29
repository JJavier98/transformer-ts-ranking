"""Official-style point metrics for the M4 benchmark.

The implementation operates on top of the local M4 loader so later runners can
evaluate model forecasts, Naive2 references and aggregate OWA scores without
duplicating metric logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.m4 import LoadedM4Dataset

__all__ = [
    "M4SeriesMetrics",
    "M4EvaluationResult",
    "evaluate_m4_dataset",
    "mase",
    "owa",
    "smape",
]


@dataclass(frozen=True)
class M4SeriesMetrics:
    """Point metrics for one M4 series.

    Attributes:
        series_id: Official M4 series identifier.
        smape: Forecast sMAPE.
        mase: Forecast MASE.
        smape_naive2: Naive2 reference sMAPE.
        mase_naive2: Naive2 reference MASE.
        owa: Forecast OWA relative to Naive2.
    """

    series_id: str
    smape: float
    mase: float
    smape_naive2: float
    mase_naive2: float
    owa: float


@dataclass(frozen=True)
class M4EvaluationResult:
    """Aggregated M4 metrics for one frequency slice.

    Attributes:
        frequency_label: Frequency evaluated.
        horizon: Official horizon for the frequency.
        seasonality: Seasonal lag used by MASE.
        series_metrics: Per-series metrics keyed by id.
        mean_smape: Mean forecast sMAPE.
        mean_mase: Mean forecast MASE.
        mean_smape_naive2: Mean Naive2 sMAPE.
        mean_mase_naive2: Mean Naive2 MASE.
        mean_owa: Aggregate OWA for the evaluated forecast set.
    """

    frequency_label: str
    horizon: int
    seasonality: int
    series_metrics: dict[str, M4SeriesMetrics]
    mean_smape: float
    mean_mase: float
    mean_smape_naive2: float
    mean_mase_naive2: float
    mean_owa: float


def smape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1e-8) -> float:
    """Compute symmetric MAPE in the standard percentage scale used by M4.

    Args:
        actual: Ground-truth future values.
        forecast: Predicted future values.
        eps: Small constant to avoid division by zero in degenerate cases.

    Returns:
        The sMAPE score in percentage units.
    """
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    forecast = np.asarray(forecast, dtype=np.float64).reshape(-1)
    if actual.shape != forecast.shape:
        raise ValueError("sMAPE requires actual and forecast arrays with the same shape.")

    denominator = np.abs(actual) + np.abs(forecast) + eps
    return float(np.mean(200.0 * np.abs(actual - forecast) / denominator))


def mase(
    insample: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    seasonality: int,
    eps: float = 1e-8,
) -> float:
    """Compute MASE using the seasonal naive scaling factor from insample data.

    Args:
        insample: Historical in-sample values.
        actual: Ground-truth future values.
        forecast: Predicted future values.
        seasonality: Seasonal lag used by the frequency.
        eps: Lower threshold used to reject zero scaling factors.

    Returns:
        The MASE score.
    """
    insample = np.asarray(insample, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    forecast = np.asarray(forecast, dtype=np.float64).reshape(-1)
    if actual.shape != forecast.shape:
        raise ValueError("MASE requires actual and forecast arrays with the same shape.")
    if len(insample) < 2:
        raise ValueError("MASE requires at least two insample observations.")

    # Tiny synthetic tests may be shorter than the official seasonal lag, so the
    # lag is clipped to keep the metric defined.
    lag = max(1, min(int(seasonality), len(insample) - 1))
    scale = np.mean(np.abs(insample[lag:] - insample[:-lag]))
    if scale <= eps:
        raise ValueError("MASE scaling factor is zero; the insample series is not usable.")
    return float(np.mean(np.abs(actual - forecast)) / scale)


def owa(smape_value: float, mase_value: float, smape_naive2: float, mase_naive2: float) -> float:
    """Compute Overall Weighted Average against the Naive2 benchmark.

    Args:
        smape_value: Forecast sMAPE.
        mase_value: Forecast MASE.
        smape_naive2: Naive2 sMAPE reference.
        mase_naive2: Naive2 MASE reference.

    Returns:
        The OWA score where values below ``1.0`` beat Naive2.
    """
    if smape_naive2 <= 0 or mase_naive2 <= 0:
        raise ValueError("OWA requires strictly positive Naive2 reference metrics.")
    return float(0.5 * ((smape_value / smape_naive2) + (mase_value / mase_naive2)))


def evaluate_m4_dataset(dataset: LoadedM4Dataset, predictions: dict[str, np.ndarray]) -> M4EvaluationResult:
    """Evaluate one frequency slice using per-series predictions keyed by id.

    Args:
        dataset: Loaded M4 frequency slice.
        predictions: Forecast arrays keyed by official M4 series id.

    Returns:
        Per-series and aggregated frequency-level metrics.
    """
    missing_predictions = [series_id for series_id in dataset.series_ids if series_id not in predictions]
    if missing_predictions:
        preview = ", ".join(missing_predictions[:3])
        raise KeyError(f"Missing predictions for M4 series: {preview}")

    seasonality = int(dataset.frequency_code)
    series_metrics: dict[str, M4SeriesMetrics] = {}

    for series_id in dataset.series_ids:
        series = dataset.series[series_id]
        forecast = np.asarray(predictions[series_id], dtype=np.float64).reshape(-1)
        if len(forecast) != dataset.horizon:
            raise ValueError(
                f"Series {series_id} forecast has horizon {len(forecast)}; expected {dataset.horizon}."
            )

        smape_value = smape(series.test_values, forecast)
        mase_value = mase(series.train_values, series.test_values, forecast, seasonality=seasonality)
        smape_naive2 = smape(series.test_values, series.naive2_forecast)
        mase_naive2 = mase(
            series.train_values,
            series.test_values,
            series.naive2_forecast,
            seasonality=seasonality,
        )
        series_metrics[series_id] = M4SeriesMetrics(
            series_id=series_id,
            smape=smape_value,
            mase=mase_value,
            smape_naive2=smape_naive2,
            mase_naive2=mase_naive2,
            owa=owa(smape_value, mase_value, smape_naive2, mase_naive2),
        )

    mean_smape = float(np.mean([metrics.smape for metrics in series_metrics.values()]))
    mean_mase = float(np.mean([metrics.mase for metrics in series_metrics.values()]))
    mean_smape_naive2 = float(np.mean([metrics.smape_naive2 for metrics in series_metrics.values()]))
    mean_mase_naive2 = float(np.mean([metrics.mase_naive2 for metrics in series_metrics.values()]))

    return M4EvaluationResult(
        frequency_label=dataset.frequency_label,
        horizon=dataset.horizon,
        seasonality=seasonality,
        series_metrics=series_metrics,
        mean_smape=mean_smape,
        mean_mase=mean_mase,
        mean_smape_naive2=mean_smape_naive2,
        mean_mase_naive2=mean_mase_naive2,
        mean_owa=owa(mean_smape, mean_mase, mean_smape_naive2, mean_mase_naive2),
    )