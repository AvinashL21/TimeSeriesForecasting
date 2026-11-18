
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(actual, predicted):
    """
    Calculate RMSE, MAE and MAPE for forecasting models.
    """

    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(actual == 0):
        raise ValueError(
            "MAPE cannot be calculated when actual values contain zero."
        )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual, predicted
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
