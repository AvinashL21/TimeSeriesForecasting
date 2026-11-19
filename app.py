import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.data_loader import load_csv_data, load_yahoo_data
from utils.preprocessing import prepare_time_series
from utils.visualization import plot_time_series
from models.arima import train_arima, fit_arima_model
from models.sarima import train_sarima, fit_sarima_model
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


def simulate_forecast_path(point_forecast, historical_values, seed=42):
    """Create one reproducible illustrative path around a model's point forecast.

    Daily log-return innovations are resampled from the selected historical
    series. The model's expected step-to-step movement is retained, while the
    resampled innovations add realistic-looking day-to-day variation. This is
    a simulated scenario, NOT the model's point forecast or a guaranteed price.
    """
    point = np.asarray(point_forecast, dtype=float).ravel()
    history = np.asarray(historical_values, dtype=float).ravel()
    if len(point) == 0:
        raise ValueError("Cannot simulate a path for an empty forecast.")
    if len(history) < 3 or np.any(history <= 0):
        raise ValueError("At least three positive historical prices are needed.")

    log_returns = np.diff(np.log(history))
    # Remove the average drift: the fitted model already supplies the expected
    # forecast direction, so only historical return innovations are resampled.
    innovations = log_returns - np.mean(log_returns)
    low, high = np.quantile(innovations, [0.01, 0.99])
    innovations = np.clip(innovations, low, high)
    if not np.isfinite(innovations).all() or np.std(innovations) == 0:
        raise ValueError("Historical returns do not provide usable variation.")

    rng = np.random.default_rng(seed)
    sampled = rng.choice(innovations, size=len(point), replace=True)
    scenario = np.empty(len(point), dtype=float)
    scenario[0] = point[0]
    for i in range(1, len(point)):
        expected_change = point[i] - point[i - 1]
        shock = np.expm1(sampled[i])
        scenario[i] = scenario[i - 1] + expected_change + scenario[i - 1] * shock

    if not np.isfinite(scenario).all():
        raise ValueError("Simulated scenario contains non-finite values.")
    return scenario


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

    # --------------------------------------------------
    # EXPLORATORY TIME SERIES ANALYSIS
    # --------------------------------------------------
    st.divider()
    st.header("Exploratory Time Series Analysis")
    st.write(
        "Analyse stationarity, autocorrelation and seasonal structure before "
        "training the forecasting models. ACF/PACF are calculated on the "
        "first-differenced closing-price series to reduce the effect of trend."
    )

    analysis_tabs = st.tabs([
        "Stationarity (ADF)",
        "ACF & PACF",
        "Autocorrelation",
        "Decomposition"
    ])

    # Stationarity: test both the original price series and first differences.
    with analysis_tabs[0]:
        st.subheader("Augmented Dickey-Fuller (ADF) Test")
        adf_series = close.dropna()
        diff_series = adf_series.diff().dropna()

        if len(adf_series) >= 20 and len(diff_series) >= 20:
            try:
                original_adf = adfuller(adf_series, autolag="AIC")
                differenced_adf = adfuller(diff_series, autolag="AIC")

                adf_col1, adf_col2 = st.columns(2)
                with adf_col1:
                    st.markdown("**Original closing-price series**")
                    st.metric("ADF statistic", f"{original_adf[0]:.4f}")
                    st.metric("p-value", f"{original_adf[1]:.6f}")
                    st.caption(
                        "Reject the unit-root null hypothesis at 5% when "
                        "the p-value is below 0.05."
                    )
                with adf_col2:
                    st.markdown("**First-differenced series**")
                    st.metric("ADF statistic", f"{differenced_adf[0]:.4f}")
                    st.metric("p-value", f"{differenced_adf[1]:.6f}")
                    st.caption(
                        "Differencing can help remove a changing level or trend."
                    )

                if original_adf[1] < 0.05:
                    st.success("The original series is stationary according to the 5% ADF test.")
                else:
                    st.warning("The original series is non-stationary according to the 5% ADF test.")

                if differenced_adf[1] < 0.05:
                    st.success("The first-differenced series is stationary according to the 5% ADF test.")
                else:
                    st.warning("The first-differenced series remains non-stationary according to the 5% ADF test.")
            except Exception as exc:
                st.error(f"ADF test could not be calculated: {exc}")
        else:
            st.info("At least 20 observations are required for the ADF analysis.")

    # ACF and PACF are complementary views of correlation across lags.
    with analysis_tabs[1]:
        st.subheader("ACF and PACF of First Differences")
        correlation_series = close.diff().dropna()
        max_lags = min(40, max(1, len(correlation_series) // 2 - 1))

        if len(correlation_series) >= 20 and max_lags >= 2:
            acf_col, pacf_col = st.columns(2)
            with acf_col:
                acf_fig, acf_ax = plt.subplots(figsize=(7, 4))
                plot_acf(
                    correlation_series,
                    lags=max_lags,
                    ax=acf_ax,
                    zero=False
                )
                acf_ax.set_title("Autocorrelation Function (ACF)")
                acf_ax.set_xlabel("Lag")
                acf_ax.set_ylabel("Autocorrelation")
                acf_fig.tight_layout()
                st.pyplot(acf_fig, use_container_width=True)
                plt.close(acf_fig)

            with pacf_col:
                pacf_fig, pacf_ax = plt.subplots(figsize=(7, 4))
                plot_pacf(
                    correlation_series,
                    lags=max_lags,
                    ax=pacf_ax,
                    method="ywm",
                    zero=False
                )
                pacf_ax.set_title("Partial Autocorrelation Function (PACF)")
                pacf_ax.set_xlabel("Lag")
                pacf_ax.set_ylabel("Partial autocorrelation")
                pacf_fig.tight_layout()
                st.pyplot(pacf_fig, use_container_width=True)
                plt.close(pacf_fig)

            st.caption(
                "Bars extending beyond the shaded confidence bounds indicate "
                "lags with statistically notable correlation at the approximate "
                "95% level. ACF shows total lag correlation; PACF shows the "
                "correlation remaining after accounting for shorter lags."
            )
        else:
            st.info("At least 20 observations are required to display ACF and PACF.")

    # A separate compact autocorrelation view: lag-1 coefficient and ACF curve.
    with analysis_tabs[2]:
        st.subheader("Autocorrelation Summary")
        autocorr_series = close.diff().dropna()
        if len(autocorr_series) >= 3:
            lag_limit = min(20, len(autocorr_series) - 1)
            lag1_corr = autocorr_series.autocorr(lag=1)
            st.metric("Lag-1 autocorrelation of first differences", f"{lag1_corr:.4f}")

            acf_summary_fig, acf_summary_ax = plt.subplots(figsize=(12, 4))
            plot_acf(
                autocorr_series,
                lags=lag_limit,
                ax=acf_summary_ax,
                zero=True
            )
            acf_summary_ax.set_title("Autocorrelation by Lag")
            acf_summary_ax.set_xlabel("Lag")
            acf_summary_ax.set_ylabel("Autocorrelation")
            acf_summary_fig.tight_layout()
            st.pyplot(acf_summary_fig, use_container_width=True)
            plt.close(acf_summary_fig)
            st.caption(
                "This is another view of the ACF: it summarises how the "
                "first-differenced series relates to its past values."
            )
        else:
            st.info("At least four observations are required for autocorrelation analysis.")

    # Classical additive decomposition on observed closing prices.
    with analysis_tabs[3]:
        st.subheader("Time Series Decomposition")
        decomposition_period = st.selectbox(
            "Seasonal period (trading observations)",
            options=[5, 21],
            index=1,
            help="5 approximates a trading week; 21 approximates a trading month."
        )
        if len(close) >= 2 * decomposition_period:
            try:
                decomposition = seasonal_decompose(
                    close,
                    model="additive",
                    period=decomposition_period,
                    extrapolate_trend="freq"
                )
                decomp_fig = decomposition.plot()
                decomp_fig.set_size_inches(12, 8)
                decomp_fig.suptitle(
                    f"Additive Decomposition (period = {decomposition_period})",
                    y=1.02
                )
                decomp_fig.tight_layout()
                st.pyplot(decomp_fig, use_container_width=True)
                plt.close(decomp_fig)
                st.caption(
                    "The decomposition separates the observed series into "
                    "trend, seasonal and residual components. The chosen period "
                    "is an analysis assumption, not proof that seasonality exists."
                )
            except Exception as exc:
                st.error(f"Time-series decomposition failed: {exc}")
        else:
            st.info(
                f"At least {2 * decomposition_period} observations are required "
                f"for decomposition with period {decomposition_period}."
            )

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
                        # Select the best candidate using training-set AIC only.
                        evaluation_results[name] = train_arima(train, test)
                    elif name == "SARIMA":
                        # Compare a compact set of seasonal/non-seasonal orders
                        # using training-set AIC only (no test-data leakage).
                        evaluation_results[name] = train_sarima(train, test)
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

        with st.expander("View selected ARIMA / SARIMA configurations"):
            for model_name in ("ARIMA", "SARIMA"):
                if model_name in evaluation_results:
                    model_output = evaluation_results[model_name]
                    if model_name == "ARIMA":
                        st.write(
                            f"**ARIMA order:** {model_output['order']} | "
                            f"**Trend:** {model_output.get('trend', 'n')} | "
                            f"**Training AIC:** {model_output['aic']:.2f}"
                        )
                    else:
                        st.write(
                            f"**SARIMA order:** {model_output['order']} | "
                            f"**Seasonal order:** {model_output['seasonal_order']} | "
                            f"**Training AIC:** {model_output['aic']:.2f}"
                        )
            st.caption(
                "Orders are selected using training data only. ARIMA compares "
                "several non-seasonal orders and trend specifications. SARIMA tests "
                "5-observation (approximate trading week) and 21-observation "
                "(approximate trading month) seasonal cycles. These are hypotheses, "
                "not guaranteed patterns in every dataset."
            )

        st.subheader("Actual vs Predicted Values")
        st.caption(
            "ARIMA and SARIMA use one-step-ahead walk-forward evaluation: each prediction uses observations available up to the previous test date. This lets the evaluation predictions respond to changing market values. The separate Future Forecast section remains a multi-step forecast."
        )
        pred_fig, pred_ax = plt.subplots(figsize=(12, 5))
        pred_ax.plot(test.index, test.values, label="Actual Prices", linewidth=2)
        # Distinct styles and sparse markers keep near-overlapping ARIMA and
        # SARIMA predictions identifiable without artificially shifting values.
        prediction_styles = {
            "ARIMA": {"color": "#ff7f0e", "linestyle": "--", "marker": "o"},
            "SARIMA": {"color": "#2ca02c", "linestyle": ":", "marker": "s"},
            "Prophet": {"color": "#d62728", "linestyle": "-", "marker": None},
            "LSTM": {"color": "#9467bd", "linestyle": "-.", "marker": None},
        }
        marker_step = max(1, len(test) // 18)
        for name, output in evaluation_results.items():
            style = prediction_styles.get(name, {})
            pred_ax.plot(
                test.index,
                np.asarray(output["predictions"]).ravel(),
                label=f"{name} Predictions",
                linewidth=2,
                alpha=0.95,
                zorder=4 if name == "ARIMA" else 3,
                markevery=marker_step if style.get("marker") else None,
                markersize=4,
                **style,
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
            "date range. Forecast dates begin after the final historical date. "
            "The chart can also show reproducible simulated paths based on "
            "resampled historical daily returns; these are illustrative scenarios, "
            "not additional model predictions or guaranteed prices."
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
                            # Refit on all observations using the configuration
                            # selected during chronological training-only evaluation.
                            # This keeps evaluation and future forecasting consistent.
                            selected = st.session_state["evaluation_results"]["ARIMA"]
                            fitted = fit_arima_model(
                                full_series,
                                order=selected["order"],
                                trend=selected.get("trend", "n"),
                            )
                            values = np.asarray(
                                fitted["model"].forecast(steps=current_horizon)
                            ).ravel()

                        elif name == "SARIMA":
                            # Preserve the training-selected seasonal specification
                            # when refitting on the complete historical series.
                            selected = st.session_state["evaluation_results"]["SARIMA"]
                            fitted = fit_sarima_model(
                                full_series,
                                order=selected["order"],
                                seasonal_order=selected["seasonal_order"],
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

            # Keep the statistical point forecasts visible, and add a
            # reproducible bootstrapped scenario path to show plausible daily
            # variation. Scenario paths are illustrative, not point estimates.
            model_colors = {
                "ARIMA": "#ff7f0e",
                "SARIMA": "#2ca02c",
                "Prophet": "#d62728",
                "LSTM": "#9467bd",
            }
            model_seeds = {"ARIMA": 101, "SARIMA": 202, "Prophet": 303, "LSTM": 404}
            for name in future_table.columns:
                color = model_colors.get(name)
                point_values = future_table[name].to_numpy(dtype=float)
                forecast_ax.plot(
                    plot_dates,
                    point_values,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.65,
                    label=f"{name} Expected Forecast"
                )
                try:
                    scenario_values = simulate_forecast_path(
                        point_values,
                        data["Close"].to_numpy(dtype=float),
                        seed=model_seeds.get(name, 42)
                    )
                    forecast_ax.plot(
                        plot_dates,
                        scenario_values,
                        color=color,
                        linestyle="-",
                        linewidth=2.2,
                        label=f"{name} Simulated Path"
                    )
                except Exception as scenario_error:
                    st.warning(f"{name} scenario path could not be generated: {scenario_error}")

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
            st.info(
                "Solid coloured lines are simulated scenario paths with "
                "resampled historical daily-return variation. Dotted lines are "
                "the models' original expected forecasts. The downloadable CSV "
                "contains the expected forecasts, not the simulated scenarios."
            )

            st.download_button(
                "Download Future Forecasts",
                data=future_table.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="future_time_series_forecasts.csv",
                mime="text/csv"
            )

except Exception as exc:
    st.error(f"Error processing dataset: {exc}")
