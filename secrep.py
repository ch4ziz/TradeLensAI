import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import openai
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging

# ----------------------
#       SETUP
# ----------------------
st.set_page_config(page_title="AI Empowered Investment Toolkit", layout="wide")

# Replace with your own OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Title and Description
st.title("📊 AI Empowered Investment Toolkit")
st.markdown(
    """
This comprehensive tool retrieves live stock data, insider trading activity, key fundamentals, latest news, and performs AI-driven sentiment & valuation analysis – **instantly**. 

> **Disclaimer**: This app is for *educational* purposes only. It is *not* financial advice. Always do your own research and consult professionals before making investment decisions.
"""
)

# ----------------------
#   HELPER FUNCTIONS
# ----------------------
@st.cache_data(ttl=3600)
def fetch_insider_trades(ticker: str) -> pd.DataFrame:
    """
    Fetch recent insider trades from OpenInsider for a given ticker.
    """
    url = f"http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=0&td=0&fdlyl=&tdlyl=&daysago=30"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="tinytable")
        if not table:
            return pd.DataFrame()

        tbody = table.find("tbody")
        rows = tbody.find_all("tr")
        all_data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 13:
                continue
            row_data = {
                "FilingDate": cols[1].get_text(strip=True),
                "TradeDate": cols[2].get_text(strip=True),
                "InsiderName": cols[5].get_text(strip=True),
                "TradeType": cols[7].get_text(strip=True),
                "SharesTraded": cols[9].get_text(strip=True),
                "Price": cols[8].get_text(strip=True),
                "Value": cols[12].get_text(strip=True),
            }
            all_data.append(row_data)
        return pd.DataFrame(all_data)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        return history
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_latest_news(ticker: str, max_news: int = 5) -> str:
    """
    Fetch the latest financial news from Finviz for a given ticker.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        if not news_table:
            return "No recent news found."
        news_items = news_table.find_all("tr")[:max_news]
        return "\n".join([item.get_text(" | ").strip() for item in news_items])
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return f"Error fetching news: {e}"

@st.cache_data(ttl=3600)
def generate_analysis_via_gpt(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Generate text using OpenAI's GPT model based on a given prompt.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt},
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Error generating GPT analysis: {e}")
        return f"Error generating GPT analysis: {e}"

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch basic fundamentals from Yahoo Finance (market cap, P/E, etc.).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Extract some key fundamentals safely
        fundamentals = {
            "Market Cap": info.get("marketCap"),
            "PE Ratio (TTM)": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "EPS (TTM)": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "52-Week High": info.get("fiftyTwoWeekHigh"),
            "52-Week Low": info.get("fiftyTwoWeekLow"),
        }
        return fundamentals
    except Exception as e:
        logging.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}

# ----------------------
#     MAIN APP FLOW
# ----------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "").upper()

if ticker:
    with st.spinner("Generating Investment Report..."):
        # Fetch data
        stock_data = fetch_stock_data(ticker)
        fundamentals = fetch_fundamentals(ticker)
        insider_trades = fetch_insider_trades(ticker)
        news = fetch_latest_news(ticker, max_news=5)

        # ----------------------
        #   PRICE CHART
        # ----------------------
        if not stock_data.empty:
            st.subheader("📈 Stock Price & Volume Trend")

            fig, ax = plt.subplots()
            ax.plot(stock_data.index, stock_data["Close"], label="Closing Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax2 = ax.twinx()
            ax2.bar(stock_data.index, stock_data["Volume"], alpha=0.3, label="Volume")
            ax2.set_ylabel("Volume")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)
        else:
            st.warning(f"Could not fetch stock data for '{ticker}'.")

        # ----------------------
        #   FUNDAMENTALS
        # ----------------------
        if fundamentals:
            st.subheader("🏦 Key Fundamentals")
            fundamentals_df = pd.DataFrame(
                list(fundamentals.items()), columns=["Metric", "Value"]
            )
            # Format large numbers like Market Cap
            def format_large_num(num):
                if num is None:
                    return "N/A"
                elif abs(num) >= 1e9:
                    return f"{num/1e9:.2f}B"
                elif abs(num) >= 1e6:
                    return f"{num/1e6:.2f}M"
                elif abs(num) >= 1e3:
                    return f"{num/1e3:.2f}K"
                else:
                    return str(round(num, 2))

            fundamentals_df["Value"] = fundamentals_df["Value"].apply(format_large_num)
            st.dataframe(fundamentals_df)
        else:
            st.warning("Could not retrieve fundamentals for this ticker.")

        # ----------------------
        #   LATEST NEWS
        # ----------------------
        st.subheader("📰 Latest Financial News")
        st.write(news)

        # ----------------------
        #   SENTIMENT ANALYSIS
        # ----------------------
        if news and "Error" not in news:
            prompt_sentiment = (
                f"Analyze the following news articles about {ticker} and provide a sentiment "
                "analysis (bullish, bearish, or neutral) with reasons:\n\n"
                f"{news}\n\n"
                "Please summarize how this might affect investor perception."
            )
            sentiment_analysis = generate_analysis_via_gpt(prompt_sentiment)
            st.subheader("📊 AI Sentiment Analysis")
            st.write(sentiment_analysis)
        else:
            st.warning("No valid news available to analyze sentiment.")

        # ----------------------
        #   VALUATION ANALYSIS
        # ----------------------
        st.subheader("💡 Optional: GPT Valuation Analysis")
        st.markdown("*(This is an experimental feature. It may take a few seconds.)*")
        if st.button("Run Valuation Analysis"):
            prompt_valuation = (
                f"Using the fundamentals and recent performance of {ticker}, "
                "provide a high-level valuation analysis. Consider metrics like "
                "Market Cap, P/E ratio, growth prospects, and any risks. "
                "Make sure to clarify that this is not financial advice."
            )
            valuation_analysis = generate_analysis_via_gpt(prompt_valuation)
            st.write(valuation_analysis)

        # ----------------------
        #   INSIDER TRADING
        # ----------------------
        st.subheader("🏛️ Insider Trading Activity")
        if not insider_trades.empty:
            st.dataframe(insider_trades)
        else:
            st.info("No recent insider trading activity found.")

        # ----------------------
        #   RISK FACTORS
        # ----------------------
        st.subheader("⚠️ Optional: GPT Risk Factors")
        st.markdown("*(This is an experimental feature. It may take a few seconds.)*")
        if st.button("Analyze Risk Factors"):
            prompt_risks = (
                f"Analyze potential risk factors for {ticker}, considering market trends, "
                "macroeconomic conditions, industry outlook, recent news, and any relevant data. "
                "Please list them in bullet points."
            )
            risk_factors = generate_analysis_via_gpt(prompt_risks)
            st.write(risk_factors)

        # ----------------------
        #   GENERATE TXT REPORT
        # ----------------------
        if st.button("📄 Generate TXT Report"):
            try:
                file_name = f"{ticker}_investment_report.txt"
                with open(file_name, "w", encoding="utf-8") as file:
                    file.write(f"AI Investment Report for {ticker}\n\n")
                    # Stock Data
                    if not stock_data.empty:
                        file.write("Recent Stock Data (Last 5 rows):\n")
                        file.write(f"{stock_data.tail(5)}\n\n")

                    # Fundamentals
                    if fundamentals:
                        file.write("Key Fundamentals:\n")
                        for k, v in fundamentals.items():
                            file.write(f"{k}: {v}\n")
                        file.write("\n")

                    # News & Sentiment
                    file.write("Latest News:\n")
                    file.write(f"{news}\n\n")
                    if news and "Error" not in news:
                        file.write("AI Sentiment Analysis:\n")
                        file.write(f"{sentiment_analysis}\n\n")

                    # Insider Trades
                    if not insider_trades.empty:
                        file.write("Insider Trading Activity:\n")
                        file.write(insider_trades.to_string())
                        file.write("\n\n")

                    st.success(f"Investment report saved as {file_name}")
            except Exception as e:
                st.error(f"Error generating TXT report: {e}")
