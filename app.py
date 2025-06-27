import streamlit as st
import requests
import pandas as pd
from prophet import Prophet
from transformers import pipeline
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# --- CONFIGURATION --- #
API_KEY = "aa7d12d1a4bfc5b669fe4f61b872c11d"

# --- PAGE TITLE --- #
st.set_page_config(page_title="Gold Investment AI", layout="wide")
st.title("📈 Gold Investment Timing AI")

# --- FETCH GOLD PRICE DATA --- #
st.subheader("Gold Price Tracker")

@st.cache_data(ttl=3600)
def fetch_gold_prices():
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365)
    url = (
        f"https://api.metalpriceapi.com/v1/timeframe?"
        f"api_key={API_KEY}&start_date={start_date}&end_date={end_date}&base=XAU&currencies=USD"
    )
    r = requests.get(url)
    data = r.json()
    if "error" in data:
        raise ValueError("API Error: " + data["error"]["message"])
    
    prices = data["rates"]
    df = pd.DataFrame([
        {"Date": date, "Close": 1 / float(info["USD"])}
        for date, info in prices.items() if "USD" in info and info["USD"] != 0
    ])
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df

try:
    gold_data = fetch_gold_prices()
    st.line_chart(gold_data.set_index("Date")["Close"], height=250)
except Exception as e:
    st.error(f"⚠️ Failed to fetch gold data: {e}")
    gold_data = pd.DataFrame()

# --- FORECASTING --- #
st.subheader("Gold Price Forecast (30 Days)")

try:
    df = gold_data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("Gold price data is empty or invalid")

    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
