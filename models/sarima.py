
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_sarima(
    train_data,
    test_data,
    order=(0, 1, 0),
    seasonal_order=(0, 0, 1, 21)
):
    """
    Train SARIMA and generate fixed-origin forecasts.
    """

    train_model = pd.Series(
        train_data.values,
        index=pd.RangeIndex(len(train_data))
    )

    model = SARIMAX(
        train_model,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    forecast = fitted_model.get_forecast(
        steps=len(test_data)
    )

    predictions = np.asarray(
        forecast.predicted_mean
    )

    rmse = np.sqrt(
        mean_squared_error(test_data.values, predictions)
    )

    mae = mean_absolute_error(
        test_data.values, predictions
    )

    mape = np.mean(
        np.abs(
            (test_data.values - predictions)
            / test_data.values
        )
    ) * 100

    return {
        'model': fitted_model,
        'predictions': predictions,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
