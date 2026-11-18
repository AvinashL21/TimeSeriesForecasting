
import pandas as pd
import yfinance as yf


def load_csv_data(file):
    """Load and validate time-series data from a CSV file."""

    try:
        df = pd.read_csv(file)

        required_columns = ['Date', 'Close']

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        df = df.dropna(subset=['Date', 'Close'])
        df = df.drop_duplicates(subset=['Date'])
        df = df.sort_values('Date')
        df = df[df['Close'] > 0]
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid observations found.")

        return df

    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")


def load_yahoo_data(ticker="^NSEI", period="5y", interval="1d"):
    """Fetch historical market data from Yahoo Finance."""

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )

        if df.empty:
            raise ValueError("No data received from Yahoo Finance.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})

        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise ValueError("Yahoo Finance data is missing Date or Close.")

        df = df[['Date', 'Close']].copy()

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        df = df.dropna(subset=['Date', 'Close'])
        df = df.drop_duplicates(subset=['Date'])
        df = df.sort_values('Date')
        df = df[df['Close'] > 0]
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid observations received.")

        return df

    except Exception as e:
        raise ValueError(f"Error fetching Yahoo Finance data: {e}")
