
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_arima(train_data, test_data, order=(0, 1, 0)):
    """
    Train ARIMA and generate fixed-origin forecasts.
    """

    train_model = pd.Series(
        train_data.values,
        index=pd.RangeIndex(len(train_data))
    )

    model = ARIMA(train_model, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(
        steps=len(test_data)
    )

    predictions = np.asarray(forecast)

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
