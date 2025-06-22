TradeLensAI - AI Investment Research Tool

Overview

This program is an AI-driven investment research tool designed to automate financial analysis by providing real-time stock data, insider trading activity, AI-generated sentiment analysis, and market insights. It streamlines the research process, offering instant analysis that would typically take financial analysts hours to compile.

Features

Stock Market Data

Retrieves real-time stock data from Yahoo Finance.

Generates price charts and volume trends.

Insider Trading Analysis

Scrapes insider trading data to reveal patterns in executive transactions.

Highlights significant sales and purchases by corporate insiders.

Fundamental Analysis

Extracts key financial metrics such as P/E Ratio, EPS, Market Cap, and Dividend Yield.

Provides quick valuation insights for informed decision-making.

Market News Aggregation

Scrapes the latest financial news related to a given stock from Finviz.

Summarizes news headlines for quick market sentiment assessment.

AI-Powered Sentiment Analysis

Uses GPT-based AI to determine bullish, bearish, or neutral sentiment.

Evaluates the potential impact of news on stock performance.

AI-Generated Risk and Valuation Analysis

Provides automated high-level stock valuation reports.

Identifies potential risk factors based on financial metrics and market trends.

Investment Report Generation

Allows users to generate and download detailed TXT reports summarizing all insights.

Installation

Clone the repository:

git clone https://github.com/reapermunky/TradeLens.git

Navigate to the project directory

Install the required dependencies:

pip install -r requirements.txt

Running the Application

Ensure you have Python installed (Python 3.7 or later is recommended).

Run the Streamlit application:

streamlit run secrep.py

Open the provided local URL in your browser to access the tool.

Deployment on Streamlit Cloud

To deploy on Streamlit Cloud:

Push your repository to GitHub.

Go to Streamlit Cloud and log in with your GitHub account.

Create a new app and select your repository.

Deploy and get a public link to share with others.

Usage Instructions

Enter a stock ticker (e.g., AAPL, TSLA, MSFT) in the input field.

The tool will retrieve and display:

Stock price trends

Insider trading activity

Key fundamental metrics

Recent financial news

AI-generated sentiment analysis

AI-generated risk assessment

Optionally, generate a TXT report summarizing the findings.

Technologies Used

Python for data processing and backend logic

Streamlit for the user interface

Yahoo Finance API (yfinance) for stock data retrieval

BeautifulSoup for web scraping insider trades and news

OpenAI GPT API for AI-driven sentiment and valuation analysis

Matplotlib for data visualization

Disclaimer

This tool is intended for educational and informational purposes only. It does not constitute financial advice. Users should conduct their own research and consult professionals before making investment decisions.

License

This project is licensed under the MIT License.

Author

Developed by C.A.
