# Time Series Forecasting and Analysis Dashboard

An interactive Streamlit application for exploring historical time-series data, comparing forecasting models, and generating future forecasts.

## Features
- Load financial time-series data from Yahoo Finance or upload a CSV.
- Explore historical prices, descriptive statistics, and daily percentage returns.
- Compare ARIMA, SARIMA, Prophet, and LSTM using a chronological train/test split.
- Evaluate forecasts using RMSE, MAE, and MAPE.
- Refit selected models on the complete selected history to generate future forecasts.
- Download filtered data and future forecast results as CSV files.

## ARIMA and SARIMA model selection
ARIMA and SARIMA use compact candidate-order searches. Candidate models are fitted on the training portion only, and the configuration with the lowest training AIC is selected before evaluating on the held-out test period. ARIMA compares several non-seasonal orders and compatible deterministic trend specifications. SARIMA compares seasonal candidates with periods of 5 observations (approximately one trading week) and 21 observations (approximately one trading month). Future forecasts repeat model selection using the complete selected historical series.

The seasonal periods are experimental modelling hypotheses, not proof that the data has a true weekly or monthly seasonal pattern. Financial market series can behave approximately like random walks, so ARIMA/SARIMA multi-step forecasts may still be relatively flat. Similar predictions are possible when the data does not contain a strong seasonal signal. A forecast should never be artificially altered merely to make model lines look different.

## CSV format
Uploaded CSV files must contain `Date` and `Close` columns. Dates should be parseable and closing prices numeric.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
The app can be deployed from this GitHub repository using Streamlit Community Cloud. Set the main file path to `app.py`.

## Limitations
- Forecasts are statistical estimates, not investment advice.
- Forecast dates are approximate weekdays and do not automatically exclude exchange holidays.
- Model performance depends on the selected dataset, date range, and train/test split.
