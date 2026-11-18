import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.data_loader import load_csv_data, load_yahoo_data
from utils.preprocessing import prepare_time_series
from utils.visualization import plot_time_series
from models.arima import train_arima
from models.sarima import train_sarima
from models.prophet_model import train_prophet
from models.lstm import train_lstm


st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Forecasting and Analysis Dashboard")
st.caption(
    "Explore historical data, examine descriptive statistics, "
    "compare forecasting models and generate future forecasts."
)


def future_business_dates(last_date, periods):
    """Approximate future trading dates by excluding weekends."""
    return pd.bdate_range(
        start=pd.Timestamp(last_date) + pd.offsets.BDay(1),
        periods=periods
    )


def recursive_lstm_forecast(model, scaler, history, periods, lookback=60):
    """Recursively forecast future values using the trained LSTM."""
    values = np.asarray(history, dtype=float).reshape(-1, 1)
    if len(values) < lookback:
        raise ValueError(
            f"LSTM requires at least {lookback} historical observations."
        )

    sequence = scaler.transform(values)[-lookback:].copy()
    forecasts_scaled = []

    for _ in range(periods):
        x = sequence.reshape(1, lookback, 1)
        next_value = float(model.predict(x, verbose=0)[0, 0])
        forecasts_scaled.append(next_value)
        sequence = np.concatenate(
            [sequence[1:], np.array([[next_value]])],
            axis=0
        )

    return scaler.inverse_transform(
        np.asarray(forecasts_scaled).reshape(-1, 1)
    ).ravel()


def render_model_comparison(results):
    rows = []
    for model_name, output in results.items():
        rows.append({
            "Model": model_name,
            "RMSE": output["rmse"],
            "MAE": output["mae"],
            "MAPE (%)": output["mape"]
        })

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    st.subheader("Model Performance Comparison")
    st.dataframe(
        comparison.style.format({
            "RMSE": "{:,.3f}",
            "MAE": "{:,.3f}",
            "MAPE (%)": "{:.3f}%"
        }),
        use_container_width=True
    )

    best = comparison.iloc[0]
    st.success(
        f"Best model on the selected test period: **{best['Model']}** "
        f"(RMSE: {best['RMSE']:,.3f})."
    )

    for metric in ["RMSE", "MAE", "MAPE (%)"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(comparison["Model"], comparison[metric])
        ax.set_title(f"Model Comparison - {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    return comparison


# --------------------------------------------------
# DATA SOURCE
# --------------------------------------------------
st.sidebar.header("Data Input")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Upload CSV", "Yahoo Finance"]
)

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df = load_csv_data(uploaded_file)
        except Exception as exc:
            st.sidebar.error(f"CSV loading failed: {exc}")
else:
    ticker = st.sidebar.text_input("Yahoo Finance ticker", value="^NSEI")
    period = st.sidebar.selectbox(
        "Historical period", ["1y", "2y", "5y", "10y"], index=2
    )
    if st.sidebar.button("Fetch Data"):
        try:
            with st.spinner("Fetching Yahoo Finance data..."):
                st.session_state["live_data"] = load_yahoo_data(
                    ticker=ticker, period=period, interval="1d"
                )
                st.session_state["live_ticker"] = ticker
                st.session_state["live_period"] = period
        except Exception as exc:
            st.sidebar.error(f"Yahoo Finance request failed: {exc}")

    if "live_data" in st.session_state:
        df = st.session_state["live_data"]

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------
if df is None:
    st.info("Select a data source and load a dataset to get started.")
    st.code("Required CSV columns: Date, Close")
    st.stop()

try:
    processed_data = prepare_time_series(df)
    if processed_data.empty:
        st.error("The selected dataset contains no valid observations.")
        st.stop()

    st.success("Dataset loaded and processed successfully!")

    st.subheader("Select Date Range")
    min_date = processed_data.index.min().date()
    max_date = processed_data.index.max().date()
    date_range = st.date_input(
        "Choose the analysis period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Please select both a start date and an end date.")
        st.stop()

    start_date, end_date = date_range
    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    data = processed_data.loc[
        (processed_data.index.date >= start_date) &
        (processed_data.index.date <= end_date)
    ].copy()

    if data.empty:
        st.warning("No observations found in the selected date range.")
        st.stop()

    # Overview
    st.divider()
    st.header("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Observations", f"{len(data):,}")
    c2.metric("Start Date", data.index.min().strftime("%d-%m-%Y"))
    c3.metric("End Date", data.index.max().strftime("%d-%m-%Y"))

    # Historical chart
    st.divider()
    st.header("Historical Time Series")
    fig = plot_time_series(data["Close"])
    st.pyplot(fig)
    plt.close(fig)

    # Descriptive statistics
    st.divider()
    st.header("Descriptive Statistical Analysis")
    close = data["Close"].astype(float)
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean Closing Price", f"₹{close.mean():,.2f}")
    s2.metric("Median Closing Price", f"₹{close.median():,.2f}")
    s3.metric("Minimum Price", f"₹{close.min():,.2f}")
    s4, s5, s6 = st.columns(3)
    s4.metric("Maximum Price", f"₹{close.max():,.2f}")
    s5.metric("Standard Deviation", f"₹{close.std():,.2f}")
    s6.metric("Price Range", f"₹{close.max() - close.min():,.2f}")

    with st.expander("View Detailed Statistics"):
        st.dataframe(
            close.describe().to_frame("Closing Price"),
            use_container_width=True
        )

    # Daily returns
    st.divider()
    st.header("Daily Percentage Returns")
    daily_returns = close.pct_change() * 100
    ret_fig, ret_ax = plt.subplots(figsize=(12, 4))
    ret_ax.plot(data.index, daily_returns, label="Daily Returns")
    ret_ax.axhline(0, linestyle="--", alpha=0.7)
    ret_ax.set_title("Daily Percentage Returns")
    ret_ax.set_xlabel("Date")
    ret_ax.set_ylabel("Return (%)")
    ret_ax.grid(True, alpha=0.3)
    ret_ax.legend()
    ret_fig.tight_layout()
    st.pyplot(ret_fig)
    plt.close(ret_fig)

    r1, r2, r3 = st.columns(3)
    r1.metric("Average Daily Return", f"{daily_returns.mean():.3f}%")
    r2.metric("Highest Daily Return", f"{daily_returns.max():.3f}%")
    r3.metric("Lowest Daily Return", f"{daily_returns.min():.3f}%")

    # Sorting and download
    st.divider()
    st.header("Dataset Preview")
    sort_by = st.selectbox(
        "Sort dataset",
        [
            "Date: Oldest to Newest",
            "Date: Newest to Oldest",
            "Closing Price: Ascending",
            "Closing Price: Descending"
        ]
    )
    preview = data.reset_index()
    if sort_by == "Date: Newest to Oldest":
        preview = preview.sort_values("Date", ascending=False)
    elif sort_by == "Closing Price: Ascending":
        preview = preview.sort_values("Close", ascending=True)
    elif sort_by == "Closing Price: Descending":
        preview = preview.sort_values("Close", ascending=False)

    st.dataframe(preview.head(20), use_container_width=True)
    st.download_button(
        "Download Filtered Dataset",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name="filtered_time_series_data.csv",
        mime="text/csv"
    )

    # --------------------------------------------------
    # MODEL TRAINING AND EVALUATION
    # --------------------------------------------------
    st.divider()
    st.header("Time Series Forecasting")
    st.write(
        "Models are evaluated with a chronological train-test split. "
        "Future forecasts are then generated by refitting each selected "
        "model on the complete selected historical series."
    )

    if len(data) < 100:
        st.warning("Select at least 100 observations to run the forecasting models.")
        st.stop()

    fc1, fc2 = st.columns(2)
    test_percent = fc1.slider(
        "Test data size (%)", 10, 30, 20, 5
    )
    horizon = fc2.selectbox(
        "Future forecast horizon (business days)",
        [5, 10, 20, 30, 60],
        index=3
    )
    selected_models = st.multiselect(
        "Select forecasting models",
        ["ARIMA", "SARIMA", "Prophet", "LSTM"],
        default=["ARIMA", "SARIMA", "Prophet", "LSTM"]
    )

    run_models = st.button(
        "Run Forecasting Models",
        type="primary",
        disabled=not selected_models
    )

    if run_models:
        series = data["Close"].astype(float).dropna()
        split = int(len(series) * (1 - test_percent / 100))
        train = series.iloc[:split]
        test = series.iloc[split:]

        if len(train) < 80 or len(test) < 10:
            st.error("Insufficient training or test observations.")
            st.stop()

        evaluation_results = {}
        model_errors = {}

        with st.spinner("Training and evaluating models..."):
            for name in selected_models:
                try:
                    if name == "ARIMA":
                        evaluation_results[name] = train_arima(
                            train, test, order=(0, 1, 0)
                        )
                    elif name == "SARIMA":
                        evaluation_results[name] = train_sarima(
                            train, test,
                            order=(0, 1, 0),
                            seasonal_order=(0, 0, 1, 21)
                        )
                    elif name == "Prophet":
                        evaluation_results[name] = train_prophet(train, test)
                    elif name == "LSTM":
                        evaluation_results[name] = train_lstm(
                            train, test, lookback=60,
                            epochs=30, batch_size=32, seed=42
                        )
                except Exception as exc:
                    model_errors[name] = str(exc)

        for name, message in model_errors.items():
            st.error(f"{name} evaluation failed: {message}")

        if evaluation_results:
            st.session_state["evaluation_results"] = evaluation_results
            st.session_state["evaluation_train"] = train
            st.session_state["evaluation_test"] = test
            st.session_state["evaluation_dates"] = test.index
            st.session_state["forecast_selected_models"] = selected_models
            st.session_state["forecast_horizon"] = horizon
            st.session_state["forecast_source_end"] = data.index[-1]

            # Clear forecasts from a previous run
            st.session_state.pop("future_forecast_table", None)

    # Persist evaluation output through Streamlit reruns
    if "evaluation_results" in st.session_state:
        evaluation_results = st.session_state["evaluation_results"]
        train = st.session_state["evaluation_train"]
        test = st.session_state["evaluation_test"]

        comparison = render_model_comparison(evaluation_results)

        st.subheader("Actual vs Predicted Values")
        pred_fig, pred_ax = plt.subplots(figsize=(12, 5))
        pred_ax.plot(test.index, test.values, label="Actual Prices", linewidth=2)
        for name, output in evaluation_results.items():
            pred_ax.plot(
                test.index,
                np.asarray(output["predictions"]).ravel(),
                label=f"{name} Predictions"
            )
        pred_ax.set_title("Actual vs Predicted Closing Prices")
        pred_ax.set_xlabel("Date")
        pred_ax.set_ylabel("Closing Price")
        pred_ax.legend()
        pred_ax.grid(True, alpha=0.3)
        pred_fig.tight_layout()
        st.pyplot(pred_fig)
        plt.close(pred_fig)

        # --------------------------------------------------
        # FUTURE FORECAST: REFIT ON THE FULL SELECTED SERIES
        # --------------------------------------------------
        st.divider()
        st.header("Future Forecast")
        st.caption(
            "Each model is refitted using all observations in the selected "
            "date range. Forecast dates begin after the final historical date."
        )

        current_horizon = st.session_state["forecast_horizon"]
        current_models = st.session_state["forecast_selected_models"]

        generate_future = st.button(
            "Generate Future Forecast",
            type="primary"
        )

        if generate_future:
            full_series = data["Close"].astype(float).dropna()
            last_actual_date = full_series.index[-1]
            future_dates = future_business_dates(last_actual_date, current_horizon)

            # Dummy one-step test series lets the existing LSTM training
            # function return its fitted model/scaler after fitting on full data.
            dummy_date = future_dates[0]
            dummy_test = pd.Series(
                [float(full_series.iloc[-1])],
                index=pd.DatetimeIndex([dummy_date])
            )

            future_values = {}
            future_errors = {}

            with st.spinner("Refitting models on all historical observations..."):
                for name in current_models:
                    try:
                        if name == "ARIMA":
                            fitted = train_arima(
                                full_series, dummy_test, order=(0, 1, 0)
                            )
                            values = np.asarray(
                                fitted["model"].forecast(steps=current_horizon)
                            ).ravel()

                        elif name == "SARIMA":
                            fitted = train_sarima(
                                full_series,
                                dummy_test,
                                order=(0, 1, 0),
                                seasonal_order=(0, 0, 1, 21)
                            )
                            values = np.asarray(
                                fitted["model"].get_forecast(
                                    steps=current_horizon
                                ).predicted_mean
                            ).ravel()

                        elif name == "Prophet":
                            fitted = train_prophet(full_series, dummy_test)
                            future_frame = pd.DataFrame({"ds": future_dates})
                            values = fitted["model"].predict(
                                future_frame
                            )["yhat"].to_numpy()

                        elif name == "LSTM":
                            fitted = train_lstm(
                                full_series,
                                dummy_test,
                                lookback=60,
                                epochs=30,
                                batch_size=32,
                                seed=42
                            )
                            values = recursive_lstm_forecast(
                                fitted["model"],
                                fitted["scaler"],
                                full_series.values,
                                current_horizon,
                                lookback=60
                            )
                        else:
                            continue

                        if len(values) != current_horizon:
                            raise ValueError("Forecast length does not match horizon.")
                        if not np.all(np.isfinite(values)):
                            raise ValueError("Forecast contains non-finite values.")

                        future_values[name] = values

                    except Exception as exc:
                        future_errors[name] = str(exc)

            for name, message in future_errors.items():
                st.warning(f"{name} future forecast failed: {message}")

            if future_values:
                future_table = pd.DataFrame(
                    future_values,
                    index=future_dates
                )
                future_table.index.name = "Forecast Date"
                st.session_state["future_forecast_table"] = future_table

        # Show stored future forecasts
        if "future_forecast_table" in st.session_state:
            future_table = st.session_state["future_forecast_table"]

            st.subheader("Forecast Values")
            st.dataframe(
                future_table.style.format("{:,.2f}"),
                use_container_width=True
            )

            # Plot historical and future values on the SAME datetime axis.
            # Forecast starts strictly after the final actual observation.
            plot_history = data["Close"].tail(120)
            plot_dates = future_table.index

            forecast_fig, forecast_ax = plt.subplots(figsize=(12, 5))
            forecast_ax.plot(
                plot_history.index,
                plot_history.values,
                label="Historical Prices",
                linewidth=2
            )

            for name in future_table.columns:
                forecast_ax.plot(
                    plot_dates,
                    future_table[name].values,
                    marker="o",
                    markersize=3,
                    linestyle="--",
                    label=f"{name} Forecast"
                )

            last_date = data.index[-1]
            forecast_ax.axvline(
                last_date,
                color="red",
                linestyle=":",
                label="Forecast Start"
            )
            forecast_ax.set_title("Historical Data and Future Forecast")
            forecast_ax.set_xlabel("Date")
            forecast_ax.set_ylabel("Closing Price")
            forecast_ax.legend()
            forecast_ax.grid(True, alpha=0.3)
            forecast_fig.autofmt_xdate()
            forecast_fig.tight_layout()
            st.pyplot(forecast_fig)
            plt.close(forecast_fig)

            st.download_button(
                "Download Future Forecasts",
                data=future_table.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="future_time_series_forecasts.csv",
                mime="text/csv"
            )

except Exception as exc:
    st.error(f"Error processing dataset: {exc}")
