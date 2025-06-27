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

forecast = None
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

    st.line_chart(forecast.set_index("ds")["yhat"].tail(60), height=250)

except Exception as e:
    st.error(f"⚠️ Unable to generate forecast: {e}")

# --- NEWS SENTIMENT --- #
st.subheader("Latest News Sentiment on Gold")

@st.cache_data(ttl=3600)
def fetch_news():
    url = "https://www.reuters.com/markets/commodities/"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    headlines = soup.find_all("h3")
    return [h.text.strip() for h in headlines if "gold" in h.text.lower()][:5]

news = fetch_news()
sentiment_pipeline = pipeline("sentiment-analysis")
scores = []

for article in news:
    result = sentiment_pipeline(article)[0]
    scores.append((article, result['label'], result['score']))

for title, label, score in scores:
    st.markdown(f"**{title}**")
    st.write(f"Sentiment: {label} (Confidence: {round(score*100, 2)}%)")

# --- MARKET SIGNAL --- #
st.subheader("📊 Market Signal")

try:
    last_price = gold_data["Close"].iloc[-1]
    predicted_price = forecast["yhat"].iloc[-1] if forecast is not None else last_price
    delta = predicted_price - last_price

    average_sentiment = sum(
        [s[2] if s[1] == 'POSITIVE' else -s[2] for s in scores]
    ) / len(scores) if scores else 0

    signal = "Hold"
    if delta > 20 and average_sentiment > 0.2:
        signal = "Strong Buy"
    elif delta > 10 and average_sentiment > 0:
        signal = "Buy"
    elif delta < -10:
        signal = "Sell"

    st.metric("Suggested Action", signal)
except Exception as e:
    st.warning(f"Could not calculate signal: {str(e)}")

# --- FOOTER --- #
st.markdown("---")
st.markdown("Built with ❤️ using MetalpriceAPI, Prophet, and BERT.")


