
import pandas as pd


def prepare_time_series(df):
    """
    Prepare time series data for forecasting.
    """

    required_columns = ['Date', 'Close']

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    data = df.copy()

    data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = pd.to_numeric(
        data['Close'], errors='coerce'
    )

    data = data.dropna(subset=['Date', 'Close'])
    data = data[data['Close'] > 0]

    data = data.drop_duplicates(subset=['Date'])
    data = data.sort_values('Date')

    data = data.set_index('Date')

    return data
