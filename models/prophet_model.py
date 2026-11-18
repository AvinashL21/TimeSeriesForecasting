
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_prophet(train_data, test_data):
    """
    Train Prophet and generate fixed-origin forecasts.
    """

    # Prepare training data
    prophet_train = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })

    prophet_test = pd.DataFrame({
        'ds': test_data.index
    })

    # Use original Prophet configuration
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    # Train model
    model.fit(prophet_train)

    # Generate forecasts
    forecast = model.predict(prophet_test)

    predictions = forecast['yhat'].values

    # Evaluate performance
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
        'model': model,
        'forecast': forecast,
        'predictions': predictions,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
