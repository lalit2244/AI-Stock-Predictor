#!/usr/bin/env python
# coding: utf-8

# In[2]:


# app.py
# 📈 Stock Predictor & Comparator — Streamlit (all features in one file)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# ------------------ Page Setup ------------------ #
st.set_page_config(page_title="📈 Stock Predictor & Comparator", layout="wide")
st.title("📈 Stock Predictor & Comparator")

st.markdown("""
Use this app to **analyze a single stock** (with indicators & forecast) or **compare multiple stocks** side by side.
""")

# ------------------ Helpers ------------------ #
def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

@st.cache_data(show_spinner=False, ttl=60*10)
def load_single(ticker: str, start: pd.Timestamp, end: pd.Timestamp, auto_adjust=True):
    df = yf.download(
        ticker,
        start=start,
        end=end + pd.Timedelta(days=1),  # include end day
        auto_adjust=auto_adjust,
        progress=False,
        threads=True
    )
    return df

@st.cache_data(show_spinner=False, ttl=60*10)
def load_multi(tickers: list, start: pd.Timestamp, end: pd.Timestamp, auto_adjust=True):
    if not tickers:
        return pd.DataFrame()
    df = yf.download(
        tickers,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=True
    )
    return df

def rsi(series: pd.Series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window=20, nstd=2):
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + nstd * std
    lower = mid - nstd * std
    return mid, upper, lower

def ensure_min_points(df: pd.DataFrame, min_points=30):
    return df if len(df) >= min_points else pd.DataFrame()

# ------------------ Sidebar ------------------ #
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["Single Stock", "Compare Multiple"], horizontal=True)

default_start = date.today() - timedelta(days=365*2)
start_date = st.sidebar.date_input("Start Date", default_start, max_value=date.today())
end_date = st.sidebar.date_input("End Date", date.today(), max_value=date.today())
auto_adjust = st.sidebar.toggle("Auto-adjust for splits/dividends", value=True)

# ------------------ SINGLE STOCK MODE ------------------ #
if mode == "Single Stock":
    st.sidebar.subheader("📌 Single Stock")
    ticker = st.sidebar.text_input("Ticker (e.g., AAPL, TSLA, INFY.BO)", "AAPL").upper()
    forecast_days = st.sidebar.slider("Forecast horizon (days)", 5, 60, 30)

    run_single = st.sidebar.button("🚀 Analyze", type="primary")

    if run_single:
        with st.spinner(f"Downloading {ticker} data..."):
            df = load_single(ticker, pd.to_datetime(start_date), pd.to_datetime(end_date), auto_adjust)
        if df is None or df.empty:
            st.error("No data returned. Check the ticker or date range.")
        else:
            df = df.copy()
            df = ensure_min_points(df, min_points=5)
            if df.empty:
                st.error("Not enough data points to analyze. Try a wider date range.")
            else:
                latest_close = to_float(df["Close"].iloc[-1], default=0.0)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest Close", f"${latest_close:.2f}")
                c2.metric("Period Days", f"{len(df)}")
                c3.metric("High", f"${to_float(df['High'].max(), 0.0):.2f}")
                c4.metric("Low", f"${to_float(df['Low'].min(), 0.0):.2f}")

                tab_price, tab_ind, tab_forecast, tab_export = st.tabs(
                    ["📉 Price & Averages", "📊 Indicators", "🔮 Forecast", "⬇️ Data & Export"]
                )

                # ----- Price & Averages ----- #
                with tab_price:
                    st.subheader(f"{ticker} — Close with SMA/EMA")
                    df["SMA20"] = df["Close"].rolling(20).mean()
                    df["SMA50"] = df["Close"].rolling(50).mean()
                    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

                    fig, ax = plt.subplots(figsize=(11, 4))
                    ax.plot(df.index, df["Close"], label="Close")
                    ax.plot(df.index, df["SMA20"], label="SMA 20")
                    ax.plot(df.index, df["SMA50"], label="SMA 50")
                    ax.plot(df.index, df["EMA20"], label="EMA 20")
                    ax.set_ylabel("Price ($)")
                    ax.legend()
                    st.pyplot(fig)

                    st.caption("Tip: SMA is the simple moving average; EMA is the exponential moving average.")

                # ----- Indicators ----- #
                with tab_ind:
                    st.subheader("RSI • MACD • Bollinger Bands")
                    rsi_vals = rsi(df["Close"], window=14)
                    macd_line, signal_line, macd_hist = macd(df["Close"])
                    bb_mid, bb_up, bb_lo = bollinger(df["Close"], window=20, nstd=2)

                    cA, cB, cC = st.columns(3)
                    cA.metric("RSI (14)", f"{to_float(rsi_vals.iloc[-1], 0.0):.2f}")
                    cB.metric("MACD", f"{to_float(macd_line.iloc[-1], 0.0):.2f}")
                    cC.metric("Signal", f"{to_float(signal_line.iloc[-1], 0.0):.2f}")

                    # Bollinger plot
                    st.write("**Bollinger Bands (20, 2σ)**")
                    fig, ax = plt.subplots(figsize=(11, 4))
                    ax.plot(df.index, df["Close"], label="Close")
                    ax.plot(df.index, bb_up, label="Upper Band", linestyle="--")
                    ax.plot(df.index, bb_mid, label="Middle (MA20)", linestyle=":")
                    ax.plot(df.index, bb_lo, label="Lower Band", linestyle="--")
                    ax.set_ylabel("Price ($)")
                    ax.legend()
                    st.pyplot(fig)

                    # RSI plot
                    st.write("**RSI (14)**")
                    fig, ax = plt.subplots(figsize=(11, 2.8))
                    ax.plot(rsi_vals.index, rsi_vals, label="RSI")
                    ax.axhline(70, linestyle="--")
                    ax.axhline(30, linestyle="--")
                    ax.set_ylabel("RSI")
                    ax.legend()
                    st.pyplot(fig)

                    # MACD plot
                    st.write("**MACD**")
                    fig, ax = plt.subplots(figsize=(11, 2.8))
                    ax.plot(macd_line.index, macd_line, label="MACD")
                    ax.plot(signal_line.index, signal_line, label="Signal")
                    ax.bar(macd_hist.index, macd_hist.values.flatten(), label="Histogram")

                    ax.legend()
                    st.pyplot(fig)

                # ----- Forecast ----- #
                with tab_forecast:
                    st.subheader(f"Linear Regression Forecast — Next {forecast_days} days")

                    # Prepare 1-D y and X indexes
                    y = df["Close"].values.ravel()  # ✅ ensure 1-D
                    X = np.arange(len(y)).reshape(-1, 1)

                    model = LinearRegression()
                    model.fit(X, y)

                    future_idx = np.arange(len(y) + forecast_days).reshape(-1, 1)
                    preds = model.predict(future_idx)

                    # Build future dates safely (pandas 2.x: use inclusive='right')
                    future_dates = pd.date_range(
                        start=df.index[-1],
                        periods=forecast_days + 1,
                        inclusive="right"
                    )
                    all_dates = df.index.append(future_dates)

                    fig, ax = plt.subplots(figsize=(11, 4))
                    ax.plot(df.index, y, label="Actual")
                    ax.plot(all_dates, preds, label="Forecast", linestyle="--")
                    ax.set_ylabel("Price ($)")
                    ax.legend()
                    st.pyplot(fig)

                    current_price = to_float(y[-1], 0.0)
                    fut_price = to_float(preds[-1], current_price)
                    delta = fut_price - current_price
                    pct = (delta / current_price * 100) if current_price else 0.0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current", f"${current_price:.2f}")
                    m2.metric("Forecast (end)", f"${fut_price:.2f}")
                    m3.metric("Δ %", f"{pct:+.2f}%")

                    st.caption("Note: This simple linear model is for education only, not financial advice.")

                # ----- Data & Export ----- #
                with tab_export:
                    st.subheader("Data Preview & Download")
                    st.dataframe(df.tail(25))
                    csv = df.to_csv().encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV",
                        csv,
                        file_name=f"{ticker}_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )

    else:
        st.info("Choose a ticker and click **Analyze** to start.")

# ------------------ MULTI STOCK COMPARISON ------------------ #
else:
    st.sidebar.subheader("📌 Compare Multiple")
    tickers = st.sidebar.multiselect(
        "Select 2–8 tickers",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX", "NVDA", "JPM", "KO", "INFY.BO"],
        default=["AAPL", "MSFT", "NVDA"]
    )
    run_multi = st.sidebar.button("🚀 Compare", type="primary")

    if run_multi:
        if len(tickers) < 2:
            st.warning("Pick at least two tickers to compare.")
        else:
            with st.spinner(f"Downloading {', '.join(tickers)}..."):
                data = load_multi(tickers, pd.to_datetime(start_date), pd.to_datetime(end_date), auto_adjust)

            if data is None or data.empty:
                st.error("No data returned. Try different tickers or date range.")
            else:
                # Normalize handling whether yf returns multi-index columns or single-level
                closes = {}
                for t in tickers:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            # yfinance multi-ticker format
                            s = data[(t, "Close")].dropna().rename(t)
                        else:
                            # If only one ticker actually came back or flat columns
                            s = data["Close"].dropna().rename(t)
                        closes[t] = s
                    except Exception:
                        pass

                if not closes:
                    st.error("Could not extract 'Close' data for the selected tickers.")
                else:
                    df_close = pd.DataFrame(closes).dropna(how="all")
                    df_close = df_close.dropna()  # align dates

                    st.subheader("Normalized Performance (Start = 100)")
                    norm = df_close / df_close.iloc[0] * 100.0
                    fig, ax = plt.subplots(figsize=(11, 4))
                    for t in norm.columns:
                        ax.plot(norm.index, norm[t], label=t)
                    ax.set_ylabel("Index (start=100)")
                    ax.legend(ncol=min(4, len(norm.columns)))
                    st.pyplot(fig)

                    # Return & volatility table
                    st.subheader("Annualized Return & Volatility")
                    ret_daily = df_close.pct_change().dropna()
                    ann_return = ret_daily.mean() * 252
                    ann_vol = ret_daily.std() * np.sqrt(252)
                    stats = pd.DataFrame({
                        "Annualized Return": ann_return,
                        "Annualized Volatility": ann_vol
                    }).sort_values("Annualized Return", ascending=False)
                    st.dataframe(stats.style.format({
                        "Annualized Return": "{:.2%}",
                        "Annualized Volatility": "{:.2%}"
                    }))

                    # Correlation heatmap
                    st.subheader("Correlation (Daily Returns)")
                    corr = ret_daily.corr()

                    fig, ax = plt.subplots(figsize=(6 + 0.3*len(corr), 5 + 0.3*len(corr)))
                    cax = ax.imshow(corr.values, interpolation="nearest")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.index)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticklabels(corr.index)
                    fig.colorbar(cax)
                    st.pyplot(fig)

                    # Download combined CSV
                    st.subheader("Download Close Prices")
                    csv_multi = df_close.to_csv().encode("utf-8")
                    st.download_button(
                        "⬇️ Download Close Prices CSV",
                        csv_multi,
                        file_name=f"compare_{'-'.join(tickers)}_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("Pick tickers and click **Compare** to see multi-stock charts & tables.")

# ------------------ Footer ------------------ #
st.markdown("---")
st.caption("Educational tool only. Not financial advice.")


# In[ ]:





# In[ ]:




