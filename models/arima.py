"""ARIMA model selection, fitting, evaluation and forecasting utilities."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# A compact candidate set keeps model selection practical for an interactive app.
ARIMA_CANDIDATES = [
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
]


def _as_series(values):
    series = pd.Series(values, dtype=float).dropna().reset_index(drop=True)
    if len(series) < 30:
        raise ValueError("ARIMA requires at least 30 valid training observations.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("ARIMA training data contains non-finite values.")
    return series


def fit_arima_model(train_data, order=None):
    """Fit ARIMA; when order is None, select a candidate using training AIC."""
    train_series = _as_series(train_data)
    candidates = [order] if order is not None else ARIMA_CANDIDATES
    best_result, best_aic, best_order = None, np.inf, None
    failures = []

    for candidate in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(
                    train_series,
                    order=candidate,
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_result, best_aic, best_order = fitted, fitted.aic, candidate
        except Exception as exc:  # continue trying the remaining candidates
            failures.append(f"{candidate}: {exc}")

    if best_result is None:
        detail = "; ".join(failures[:2])
        raise RuntimeError(f"Unable to fit any ARIMA candidate. {detail}")

    return {
        "model": best_result,
        "order": best_order,
        "aic": float(best_aic),
        "trend": "n",
    }


def train_arima(train_data, test_data, order=None):
    """Select and fit ARIMA on training data, then make fixed-origin test forecasts."""
    fitted_info = fit_arima_model(train_data, order=order)
    predictions = np.asarray(
        fitted_info["model"].forecast(steps=len(test_data)), dtype=float
    ).ravel()
    actual = np.asarray(test_data, dtype=float).ravel()
    if len(predictions) != len(actual) or not np.isfinite(predictions).all():
        raise ValueError("ARIMA produced invalid or incorrectly sized predictions.")

    return {
        **fitted_info,
        "predictions": predictions,
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mae": float(mean_absolute_error(actual, predictions)),
        "mape": float(np.mean(np.abs((actual - predictions) / np.maximum(np.abs(actual), 1e-8))) * 100),
    }
