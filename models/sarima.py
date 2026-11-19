"""SARIMA model selection, fitting, evaluation and forecasting utilities."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# A small, explicit candidate set. The 21-observation seasonal period is an
# approximate trading-month hypothesis, not an assertion of true seasonality.
SARIMA_CANDIDATES = [
    ((0, 1, 1), (0, 0, 1, 21)),
    ((0, 1, 1), (1, 0, 0, 21)),
    ((0, 1, 1), (1, 0, 1, 21)),
    ((1, 1, 0), (0, 0, 1, 21)),
    ((1, 1, 0), (1, 0, 0, 21)),
    ((1, 1, 0), (1, 0, 1, 21)),
]


def _as_series(values):
    series = pd.Series(values, dtype=float).dropna().reset_index(drop=True)
    if len(series) < 60:
        raise ValueError("SARIMA requires at least 60 valid training observations.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("SARIMA training data contains non-finite values.")
    return series


def fit_sarima_model(train_data, order=None, seasonal_order=None):
    """Fit SARIMA; optionally select an order pair using training AIC."""
    train_series = _as_series(train_data)
    if order is not None and seasonal_order is not None:
        candidates = [(order, seasonal_order)]
    elif order is not None or seasonal_order is not None:
        raise ValueError("Provide both order and seasonal_order, or neither for automatic selection.")
    else:
        candidates = SARIMA_CANDIDATES

    best_result, best_aic, best_pair = None, np.inf, None
    failures = []
    for candidate_order, candidate_seasonal in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = SARIMAX(
                    train_series,
                    order=candidate_order,
                    seasonal_order=candidate_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=150)
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_result = fitted
                best_aic = fitted.aic
                best_pair = (candidate_order, candidate_seasonal)
        except Exception as exc:
            failures.append(f"{candidate_order}/{candidate_seasonal}: {exc}")

    if best_result is None:
        detail = "; ".join(failures[:2])
        raise RuntimeError(f"Unable to fit any SARIMA candidate. {detail}")

    return {
        "model": best_result,
        "order": best_pair[0],
        "seasonal_order": best_pair[1],
        "aic": float(best_aic),
    }


def train_sarima(train_data, test_data, order=None, seasonal_order=None):
    """Select and fit SARIMA on training data, then make fixed-origin forecasts."""
    fitted_info = fit_sarima_model(
        train_data, order=order, seasonal_order=seasonal_order
    )
    predictions = np.asarray(
        fitted_info["model"].get_forecast(steps=len(test_data)).predicted_mean,
        dtype=float,
    ).ravel()
    actual = np.asarray(test_data, dtype=float).ravel()
    if len(predictions) != len(actual) or not np.isfinite(predictions).all():
        raise ValueError("SARIMA produced invalid or incorrectly sized predictions.")

    return {
        **fitted_info,
        "predictions": predictions,
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mae": float(mean_absolute_error(actual, predictions)),
        "mape": float(np.mean(np.abs((actual - predictions) / np.maximum(np.abs(actual), 1e-8))) * 100),
    }
