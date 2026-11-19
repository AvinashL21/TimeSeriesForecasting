"""Prophet model fitting and one-step-ahead walk-forward evaluation."""

import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error


def fit_prophet_model(history):
    """Fit Prophet to all supplied observed values."""
    series = pd.Series(history, dtype=float).dropna()
    if len(series) < 30:
        raise ValueError("Prophet requires at least 30 valid observations.")

    frame = pd.DataFrame({
        "ds": pd.to_datetime(series.index),
        "y": series.to_numpy(dtype=float),
    })
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        n_changepoints=min(15, max(5, len(frame) // 20)),
        uncertainty_samples=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame)
    return model


def train_prophet(train_data, test_data):
    """Evaluate Prophet with expanding-window one-step-ahead predictions.

    Each test date is predicted using only observations available before that
    date. Prophet is refitted after each observation so the line can react to
    newly observed changes, matching the walk-forward protocol used by ARIMA
    and SARIMA.
    """
    history = pd.Series(train_data, dtype=float).dropna().copy()
    actual = np.asarray(test_data, dtype=float).ravel()
    predictions = []
    final_model = None

    if len(history) < 30:
        raise ValueError("Prophet requires at least 30 training observations.")

    for date, observed in zip(test_data.index, actual):
        model = fit_prophet_model(history)
        future = pd.DataFrame({"ds": [pd.Timestamp(date)]})
        prediction = float(model.predict(future)["yhat"].iloc[0])
        predictions.append(prediction)
        history.loc[pd.Timestamp(date)] = float(observed)
        final_model = model

    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Prophet produced non-finite walk-forward predictions.")

    denominator = np.maximum(np.abs(actual), 1e-8)
    return {
        "model": final_model,
        "predictions": predictions,
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mae": float(mean_absolute_error(actual, predictions)),
        "mape": float(np.mean(np.abs((actual - predictions) / denominator)) * 100),
        "evaluation_method": "one-step-ahead walk-forward",
    }
