# First, make sure to install Streamlit and yfinance
# !pip install streamlit yfinance

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# Title and description of the app
st.title("Stock Data Viewer")
st.write("This app retrieves and displays stock data from Yahoo Finance.")

# Sidebar for user input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Retrieve stock data
try:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if not stock_data.empty:
        # Display the stock data as a table
        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(stock_data)

        # Display the stock closing price chart
        st.subheader("Closing Price Chart")
        st.line_chart(stock_data['Close'])

        # Display the stock volume chart
        st.subheader("Volume Chart")
        st.line_chart(stock_data['Volume'])

    else:
        st.error("No data found. Please check the ticker or date range.")

except Exception as e:
    st.error(f"An error occurred: {e}")

