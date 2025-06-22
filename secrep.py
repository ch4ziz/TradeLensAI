import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import openai
import plotly.express as px # New import for interactive charts
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# ----------------------
#        SETUP
# ----------------------
st.set_page_config(page_title="AI Empowered Investment Toolkit", layout="wide", initial_sidebar_state="collapsed")

# Replace with your own OpenAI API key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found. Please set it in Streamlit Secrets.")
    st.stop() # Stop the app if API key is not available

# Title and Description
st.title("📊 AI Empowered Investment Toolkit")
st.markdown(
    """
This comprehensive tool retrieves live stock data, insider trading activity, key fundamentals, latest news, and performs AI-driven sentiment & valuation analysis – **instantly**. 

> **Disclaimer**: This app is for *educational* purposes only. It is *not* financial advice. Always do your own research and consult professionals before making investment decisions.
"""
)

# ----------------------
#    HELPER FUNCTIONS
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
            logging.info(f"No tinytable found for {ticker} on OpenInsider.")
            return pd.DataFrame()

        tbody = table.find("tbody")
        if not tbody:
            logging.info(f"No tbody found for {ticker} in OpenInsider table.")
            return pd.DataFrame()

        rows = tbody.find_all("tr")
        all_data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 13: # Ensure enough columns exist
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
        
        df = pd.DataFrame(all_data)
        # Convert numeric columns to appropriate types, handling errors
        df['SharesTraded'] = pd.to_numeric(df['SharesTraded'].str.replace(',', '').str.replace('+', ''), errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'].str.replace('$', ''), errors='coerce')

        return df
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching insider trades for {ticker}.")
        st.warning(f"Could not fetch insider trades for '{ticker}': Request timed out.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as req_e:
        logging.error(f"Request error fetching insider trades for {ticker}: {req_e}")
        st.warning(f"Could not fetch insider trades for '{ticker}': Network error or server issue.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching insider trades for {ticker}: {e}")
        st.warning(f"An unexpected error occurred while fetching insider trades for '{ticker}'.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        if history.empty:
            st.warning(f"No stock data found for '{ticker}'. Please check the ticker symbol.")
        return history
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        st.error(f"Could not fetch stock data for '{ticker}'. Please check the ticker symbol or try again later.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_latest_news(ticker: str, max_news: int = 5) -> str:
    """
    Fetch the latest financial news from Finviz for a given ticker.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        if not news_table:
            return "No recent news found from Finviz."
        news_items = news_table.find_all("tr")[:max_news]
        
        # Format news with time, date, and link
        formatted_news = []
        for item in news_items:
            cols = item.find_all("td")
            if len(cols) >= 2:
                time_str = cols[0].get_text(strip=True)
                news_link = cols[1].find("a")
                if news_link:
                    title = news_link.get_text(strip=True)
                    href = news_link['href']
                    formatted_news.append(f"{time_str} | [{title}]({href})")
                else:
                    formatted_news.append(item.get_text(" | ").strip()) # Fallback

        return "\n".join(formatted_news)
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching news for {ticker}.")
        st.warning(f"Could not fetch news for '{ticker}': Request timed out.")
        return "Error fetching news: Request timed out."
    except requests.exceptions.RequestException as req_e:
        logging.error(f"Request error fetching news for {ticker}: {req_e}")
        st.warning(f"Could not fetch news for '{ticker}': Network error or server issue.")
        return "Error fetching news: Network error or server issue."
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        st.warning(f"An unexpected error occurred while fetching news for '{ticker}'.")
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
                {"role": "system", "content": "You are a concise and objective financial analyst. Always include a disclaimer that your analysis is for educational purposes only and not financial advice."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3 # Lower temperature for more factual/less creative responses
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.error.AuthenticationError:
        logging.error("OpenAI API key is invalid or missing.")
        return "Error: OpenAI API key is invalid or missing. Please check your Streamlit secrets."
    except openai.error.OpenAIError as openai_e:
        logging.error(f"OpenAI API error: {openai_e}")
        return f"Error communicating with OpenAI API: {openai_e}. Please try again later."
    except Exception as e:
        logging.error(f"Error generating GPT analysis: {e}")
        return f"An unexpected error occurred during AI analysis: {e}"

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch basic fundamentals from Yahoo Finance (market cap, P/E, etc.).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            st.warning(f"Could not retrieve detailed fundamentals for '{ticker}'.")
            return {}

        # Extract some key fundamentals safely
        fundamentals = {
            "Current Price": info.get("currentPrice"),
            "Market Cap": info.get("marketCap"),
            "PE Ratio (TTM)": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "EPS (TTM)": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "52-Week High": info.get("fiftyTwoWeekHigh"),
            "52-Week Low": info.get("fiftyTwoWeekLow"),
            "Revenue (TTM)": info.get("revenueTTM"),
            "Profit Margins": info.get("profitMargins")
        }
        return {k: v for k, v in fundamentals.items() if v is not None} # Filter out None values
    except Exception as e:
        logging.error(f"Error fetching fundamentals for {ticker}: {e}")
        st.warning(f"An error occurred while fetching fundamentals for '{ticker}'.")
        return {}

# ----------------------
#       MAIN APP FLOW
# ----------------------

# Cache clear button (positioned before ticker input for immediate effect)
if st.button("🔄 Clear All Cached Data & Refresh"):
    st.cache_data.clear()
    st.rerun() # Rerun the app to fetch fresh data

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "").upper()

if ticker:
    with st.spinner(f"Generating Investment Report for {ticker}..."):
        # Fetch data
        stock_data = fetch_stock_data(ticker)
        fundamentals = fetch_fundamentals(ticker)
        insider_trades = fetch_insider_trades(ticker)
        news = fetch_latest_news(ticker, max_news=5)

        # ----------------------
        #    PRICE CHART
        # ----------------------
        if not stock_data.empty:
            st.subheader("📈 Stock Price & Volume Trend (Past 6 Months)")
            
            # Plotly Express for interactive chart
            fig_price = px.line(stock_data, x=stock_data.index, y="Close", 
                                title=f"{ticker} Closing Price",
                                labels={"Close": "Price (USD)", "index": "Date"})
            fig_price.update_xaxes(rangeslider_visible=True)

            fig_volume = px.bar(stock_data, x=stock_data.index, y="Volume", 
                                title=f"{ticker} Daily Volume",
                                labels={"Volume": "Volume", "index": "Date"},
                                opacity=0.3)
            
            # Combine charts (Plotly doesn't directly combine like matplotlib's twinx easily in px, often done with graph_objects)
            # For simplicity for this demo, we'll show them separately for now, or use a single combined Plotly go object.
            # A more advanced approach would be to use plotly.graph_objects.make_subplots
            st.plotly_chart(fig_price, use_container_width=True)
            st.plotly_chart(fig_volume, use_container_width=True)

        else:
            st.warning(f"Could not display chart as no stock data found for '{ticker}'.")

        # ----------------------
        #    FUNDAMENTALS
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
                elif isinstance(num, (int, float)):
                    if abs(num) >= 1e9:
                        return f"${num/1e9:,.2f}B"
                    elif abs(num) >= 1e6:
                        return f"${num/1e6:,.2f}M"
                    elif abs(num) >= 1e3:
                        return f"${num/1e3:,.2f}K"
                    elif 0 < abs(num) < 1:
                        return f"{num:.4f}" # For small decimals like dividend yields, profit margins
                    else:
                        return f"{num:,.2f}" # Standard number formatting
                else:
                    return str(num)

            fundamentals_df["Value"] = fundamentals_df["Value"].apply(format_large_num)
            st.dataframe(fundamentals_df.set_index("Metric")) # Set metric as index for better display
        else:
            st.info("No detailed fundamentals available for this ticker.")

        # ----------------------
        #    LATEST NEWS
        # ----------------------
        st.subheader("📰 Latest Financial News")
        if news and "No recent news found" not in news and "Error" not in news:
            st.markdown(news) # Use markdown to render links if present
        else:
            st.info(news) # Display specific error message or "No news found"

        # ----------------------
        #    AI Analysis Section (using expanders for cleaner UI)
        # ----------------------
        st.subheader("🧠 AI-Powered Insights")

        with st.expander("📊 AI Sentiment Analysis (Click to expand)"):
            if news and "No recent news found" not in news and "Error" not in news:
                prompt_sentiment = (
                    f"Analyze the following recent news articles about {ticker} and provide a concise sentiment "
                    "analysis (primarily bullish, bearish, or neutral, or mixed) with brief, objective reasons "
                    "based *only* on the provided news. Focus on how this news might affect investor perception. "
                    "Always include a disclaimer that this analysis is for educational purposes only and not financial advice.\n\n"
                    f"News Articles:\n{news}\n\n"
                    "Sentiment Analysis Summary:"
                )
                sentiment_analysis = generate_analysis_via_gpt(prompt_sentiment)
                st.write(sentiment_analysis)
            else:
                st.info("No valid news available to perform sentiment analysis.")

        with st.expander("💡 AI Valuation Analysis (Click to expand)"):
            st.markdown("*(This is an experimental feature. It may take a few seconds.)*")
            if st.button("Run Valuation Analysis", key="valuation_btn"): # Unique key for button
                if fundamentals or not stock_data.empty:
                    prompt_valuation = (
                        f"Given the available fundamentals ({fundamentals if fundamentals else 'N/A'}) and recent stock performance "
                        f"(last 5 closing prices: {stock_data['Close'].tail(5).to_dict() if not stock_data.empty else 'N/A'}) for {ticker}, "
                        "provide a high-level, concise valuation analysis. Consider metrics like "
                        "Market Cap, P/E ratio, and general market conditions. Do NOT invent data. "
                        "Make sure to clarify that this is for educational purposes only and not financial advice."
                    )
                    valuation_analysis = generate_analysis_via_gpt(prompt_valuation)
                    st.write(valuation_analysis)
                else:
                    st.warning("Not enough data to perform a meaningful valuation analysis.")

        with st.expander("⚠️ AI Risk Factors (Click to expand)"):
            st.markdown("*(This is an experimental feature. It may take a few seconds.)*")
            if st.button("Analyze Risk Factors", key="risk_btn"): # Unique key for button
                if fundamentals or news:
                    prompt_risks = (
                        f"Analyze potential risk factors for {ticker}, considering available fundamentals ({fundamentals if fundamentals else 'N/A'}), "
                        f"recent news ({news if news and 'No recent news' not in news and 'Error' not in news else 'N/A'}), and general market and industry outlook. "
                        "Please list them in concise bullet points. "
                        "Always include a disclaimer that this analysis is for educational purposes only and not financial advice."
                    )
                    risk_factors = generate_analysis_via_gpt(prompt_risks)
                    st.write(risk_factors)
                else:
                    st.warning("Not enough data to analyze risk factors.")


        # ----------------------
        #    INSIDER TRADING
        # ----------------------
        st.subheader("🏛️ Insider Trading Activity (Last 30 Days)")
        if not insider_trades.empty:
            st.dataframe(insider_trades.style.format({
                "SharesTraded": "{:,.0f}",
                "Price": "${:,.2f}",
                "Value": "${:,.0f}"
            }))
        else:
            st.info("No recent insider trading activity found or data could not be fetched.")

        # ----------------------
        #    GENERATE TXT REPORT
        # ----------------------
        st.markdown("---") # Separator
        st.subheader("📄 Generate Full Report")
        if st.button("Download Full Text Report"):
            try:
                report_content = []
                report_content.append(f"AI Investment Report for {ticker}\n")
                report_content.append(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                report_content.append("--- Stock Price & Volume ---\n")
                if not stock_data.empty:
                    report_content.append("Recent Stock Data (Last 5 rows):\n")
                    report_content.append(f"{stock_data.tail(5).to_string()}\n\n")
                else:
                    report_content.append("No stock data available.\n\n")

                report_content.append("--- Key Fundamentals ---\n")
                if fundamentals:
                    for k, v in fundamentals.items():
                        formatted_v = format_large_num(v) # Use the same formatter
                        report_content.append(f"{k}: {formatted_v}\n")
                    report_content.append("\n")
                else:
                    report_content.append("No fundamentals available.\n\n")

                report_content.append("--- Latest News ---\n")
                if news and "No recent news found" not in news and "Error" not in news:
                    report_content.append(f"{news}\n\n")
                else:
                    report_content.append(f"{news}\n\n") # Will include error/no news message

                if 'sentiment_analysis' in locals(): # Only include if analysis was run
                    report_content.append("--- AI Sentiment Analysis ---\n")
                    report_content.append(f"{sentiment_analysis}\n\n")
                
                if 'valuation_analysis' in locals(): # Only include if analysis was run
                    report_content.append("--- AI Valuation Analysis ---\n")
                    report_content.append(f"{valuation_analysis}\n\n")

                if 'risk_factors' in locals(): # Only include if analysis was run
                    report_content.append("--- AI Risk Factors ---\n")
                    report_content.append(f"{risk_factors}\n\n")

                report_content.append("--- Insider Trading Activity ---\n")
                if not insider_trades.empty:
                    report_content.append(insider_trades.to_string())
                    report_content.append("\n\n")
                else:
                    report_content.append("No recent insider trading activity found.\n\n")

                # Add a final disclaimer to the report
                report_content.append("--------------------------------------------------------------------------------\n")
                report_content.append("Disclaimer: This report is for educational purposes only. It is not financial advice. \n")
                report_content.append("Always do your own research and consult professionals before making investment decisions.\n")
                report_content.append("--------------------------------------------------------------------------------\n")


                # Use st.download_button for file download directly in browser
                st.download_button(
                    label="Click to Download Report",
                    data="".join(report_content).encode("utf-8"),
                    file_name=f"{ticker}_investment_report.txt",
                    mime="text/plain"
                )
                st.success("Report generated and ready for download!")
            except Exception as e:
                st.error(f"Error generating TXT report: {e}")
