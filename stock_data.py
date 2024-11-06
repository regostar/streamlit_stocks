import streamlit as st
import yfinance as yf

def show_stock_data():
    # Check if a stock has been selected
    if 'selected_stock' not in st.session_state:
        st.write("Please select a stock from the Stock List page.")
        return

    # Get selected stock symbol
    symbol = st.session_state.selected_stock
    ticker = yf.Ticker(symbol)

    st.title(f"Details for {symbol}")

    # Fetch and display stock information
    info = ticker.info
    st.write("**Name**:", info.get("longName", "N/A"))
    st.write("**Industry**:", info.get("industry", "N/A"))
    st.write("**Market Cap**:", info.get("marketCap", "N/A"))
    st.write("**P/E Ratio**:", info.get("trailingPE", "N/A"))
    st.write("**52-Week High**:", info.get("fiftyTwoWeekHigh", "N/A"))
    st.write("**52-Week Low**:", info.get("fiftyTwoWeekLow", "N/A"))

    # Display recent price chart
    st.subheader("Price History")
    history_data = ticker.history(period="1y")
    st.line_chart(history_data['Close'])
