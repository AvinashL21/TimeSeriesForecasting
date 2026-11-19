"""ARIMA model selection, fitting, evaluation and forecasting utilities.

Candidate models are ranked by AIC using training observations only. The
selection includes both stationary and first-differenced specifications, and
allows a deterministic trend where supported by the integration order.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# (order, trend). For d=0, a constant may be considered; for d=1,
# a linear trend represents drift in the differenced process.
ARIMA_CANDIDATES = [
    ((0, 0, 1), "c"),
    ((1, 0, 0), "c"),
    ((1, 0, 1), "c"),
    ((2, 0, 1), "c"),
    ((0, 1, 1), "n"),
    ((1, 1, 0), "n"),
    ((1, 1, 1), "n"),
    ((2, 1, 0), "n"),
    ((0, 1, 1), "t"),
    ((1, 1, 0), "t"),
    ((1, 1, 1), "t"),
]


def _as_series(values):
    series = pd.Series(values, dtype=float).dropna().reset_index(drop=True)
    if len(series) < 30:
        raise ValueError("ARIMA requires at least 30 valid training observations.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("ARIMA training data contains non-finite values.")
    return series


def fit_arima_model(train_data, order=None, trend=None):
    """Fit ARIMA; select order/trend by training AIC when not explicitly supplied."""
    train_series = _as_series(train_data)

    if order is not None:
        if trend is None:
            trend = "n" if order[1] > 0 else "c"
        candidates = [(order, trend)]
    else:
        candidates = ARIMA_CANDIDATES

    best_result, best_aic, best_order, best_trend = None, np.inf, None, None
    failures = []

    for candidate_order, candidate_trend in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(
                    train_series,
                    order=candidate_order,
                    trend=candidate_trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_result = fitted
                best_aic = fitted.aic
                best_order = candidate_order
                best_trend = candidate_trend
        except Exception as exc:
            failures.append(f"{candidate_order}, trend={candidate_trend}: {exc}")

    if best_result is None:
        detail = "; ".join(failures[:3])
        raise RuntimeError(f"Unable to fit any ARIMA candidate. {detail}")

    return {
        "model": best_result,
        "order": best_order,
        "trend": best_trend,
        "aic": float(best_aic),
    }


def train_arima(train_data, test_data, order=None, trend=None):
    """Evaluate ARIMA with expanding-history, one-step-ahead walk-forward forecasts.

    At each test date, the model predicts one step ahead, then the newly observed
    actual value is appended before predicting the next date. This is a valid
    rolling evaluation protocol and produces a time-varying prediction line.
    """
    fitted_info = fit_arima_model(train_data, order=order, trend=trend)
    rolling_result = fitted_info["model"]
    actual = np.asarray(test_data, dtype=float).ravel()
    predictions = []

    for observed_value in actual:
        next_prediction = float(
            np.asarray(rolling_result.forecast(steps=1), dtype=float).ravel()[0]
        )
        predictions.append(next_prediction)
        rolling_result = rolling_result.append(
            np.asarray([observed_value], dtype=float), refit=False
        )

    predictions = np.asarray(predictions, dtype=float)

    if len(predictions) != len(actual) or not np.isfinite(predictions).all():
        raise ValueError("ARIMA produced invalid or incorrectly sized predictions.")

    denominator = np.maximum(np.abs(actual), 1e-8)
    return {
        **fitted_info,
        "model": rolling_result,
        "predictions": predictions,
        "evaluation_method": "one-step-ahead walk-forward",
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mae": float(mean_absolute_error(actual, predictions)),
        "mape": float(np.mean(np.abs((actual - predictions) / denominator)) * 100),
    }
