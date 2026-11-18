
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_time_series(data):
    """Plot the historical time series."""

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(data.index, data.values, label='Closing Price')

    ax.set_title('Historical Time Series')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(actual, predicted, dates, model_name):
    """Plot actual versus predicted values."""

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, actual, label='Actual Prices')
    ax.plot(dates, predicted, label='Predicted Prices')

    ax.set_title(f'{model_name} - Actual vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_model_comparison(comparison_df, metric):
    """Compare forecasting models using a selected metric."""

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(comparison_df['Model'], comparison_df[metric])

    ax.set_title(f'Model Comparison - {metric}')
    ax.set_xlabel('Forecasting Models')
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    return fig


def plot_future_forecast(
    historical_data,
    forecast,
    confidence_intervals=None,
    historical_points=120
):
    """Plot historical prices and future forecasts."""

    historical_values = historical_data.iloc[-historical_points:].values

    historical_x = np.arange(len(historical_values))

    forecast_x = np.arange(
        len(historical_values),
        len(historical_values) + len(forecast)
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        historical_x,
        historical_values,
        label='Historical Prices'
    )

    ax.plot(
        forecast_x,
        np.asarray(forecast),
        label='Future Forecast',
        color='green'
    )

    if confidence_intervals is not None:
        lower = np.asarray(confidence_intervals.iloc[:, 0])
        upper = np.asarray(confidence_intervals.iloc[:, 1])

        ax.fill_between(
            forecast_x,
            lower,
            upper,
            alpha=0.2,
            label='95% Confidence Interval'
        )

    ax.axvline(
        len(historical_values) - 1,
        color='red',
        linestyle='--',
        label='Forecast Start'
    )

    ax.set_title('Historical Data and Future Forecast')
    ax.set_xlabel('Trading Observations')
    ax.set_ylabel('Closing Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
