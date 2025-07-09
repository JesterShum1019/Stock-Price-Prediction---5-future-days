import streamlit as st

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction / Analysis")
st.write("Enter a stock ticker below to get a predicted price chart.")

import yfinance as yf
import pandas as pd
from datetime import datetime

ticker = input("Enter a stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
if not ticker.isalpha():
    raise ValueError("Invalid ticker symbol. Please enter only alphabetic characters.")

end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Download stock data
stock_data = yf.download(ticker, start='2019-10-01', end=end_date, auto_adjust=False)