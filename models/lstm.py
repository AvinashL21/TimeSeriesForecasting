
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def train_lstm(
    train_data,
    test_data,
    lookback=60,
    epochs=30,
    batch_size=32,
    seed=42
):
    """
    Train LSTM and generate fixed-origin recursive forecasts.
    """

    # Prepare training and testing values
    train_values = train_data.values.reshape(-1, 1)
    test_values = test_data.values.reshape(-1, 1)

    # Scale data using training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)

    # Create training sequences
    X_train, y_train = [], []

    for i in range(lookback, len(train_scaled)):
        X_train.append(train_scaled[i-lookback:i, 0])
        y_train.append(train_scaled[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = X_train.reshape(
        X_train.shape[0], lookback, 1
    )

    # Set random seeds for reproducibility
    tf.keras.utils.set_random_seed(seed)

    # Build LSTM model
    model = Sequential([
        LSTM(
            50,
            return_sequences=True,
            input_shape=(lookback, 1)
        ),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1
    )

    # One-step-ahead walk-forward evaluation. For each test date, predict
    # using the previous 60 observed values, then append the actual observation
    # before predicting the next date. This is comparable to ARIMA/SARIMA.
    input_sequence = train_scaled[-lookback:].copy()
    predictions_scaled = []

    for actual_value in test_values:
        input_data = input_sequence.reshape(1, lookback, 1)
        next_prediction = float(model.predict(input_data, verbose=0)[0, 0])
        predictions_scaled.append(next_prediction)

        actual_scaled = scaler.transform(
            np.asarray([[float(actual_value[0])]], dtype=float)
        )[0, 0]
        input_sequence = np.concatenate(
            [input_sequence[1:], np.asarray([[actual_scaled]])],
            axis=0
        )

    # Convert predictions to original price scale
    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()

    # Evaluate performance
    rmse = np.sqrt(
        mean_squared_error(test_values, predictions)
    )

    mae = mean_absolute_error(
        test_values, predictions
    )

    mape = np.mean(
        np.abs((test_values.flatten() - predictions)
               / test_values.flatten())
    ) * 100

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'predictions': predictions,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
