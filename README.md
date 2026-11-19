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
ARIMA and SARIMA use compact candidate-order searches. Candidate models are fitted on the training portion only, and the configuration with the lowest training AIC is selected before evaluating on the held-out test period. Future forecasts repeat model selection using the complete selected historical series.

The SARIMA search uses a seasonal period of 21 observations as an approximate trading-month assumption. This is an experimental modelling choice, not proof that the data has a true 21-day seasonal pattern. Financial market series can behave approximately like random walks, so ARIMA/SARIMA multi-step forecasts may still be relatively flat. A flat forecast should not be artificially altered for visual effect.

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
