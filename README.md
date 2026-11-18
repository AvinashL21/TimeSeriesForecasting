# Live Time Series Forecasting and Analysis Dashboard

A Streamlit application for time-series exploration, model evaluation, and future forecasting.

## Features
- Load historical NIFTY 50 data from Yahoo Finance or upload a CSV file
- Explore historical prices and descriptive statistics
- Train and compare ARIMA, SARIMA, Prophet, and LSTM models
- Evaluate forecasts using RMSE, MAE, and MAPE
- Generate and download future forecasts

## CSV format
Uploaded CSV files should contain a date column and a numeric closing-price column.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push this repository to GitHub, then create an app in Streamlit Community Cloud using `app.py` as the entrypoint.
