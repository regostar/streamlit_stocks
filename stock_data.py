# stock_data.py

import streamlit as st
import yfinance as yf
import datetime

def show_stock_data():
    st.title("Stock Data Viewer")
    st.write("This page displays historical stock data.")

    # Sidebar inputs for stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    # Retrieve stock data
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            st.subheader(f"Stock Data for {ticker}")
            st.dataframe(stock_data)

            # Display the closing price and volume charts
            st.subheader("Closing Price Chart")
            st.line_chart(stock_data['Close'])

            st.subheader("Volume Chart")
            st.line_chart(stock_data['Volume'])
        else:
            st.error("No data found. Please check the ticker or date range.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
