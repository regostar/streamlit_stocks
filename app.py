import streamlit as st
from stock_data import show_stock_data
from forecast import show_forecast
from stock_list import show_stock_list  # Import the new page

# Initialize the selected page in session state
if 'page' not in st.session_state:
    st.session_state.page = "Stock List"

# Display the appropriate page based on session state
if st.session_state.page == "Stock List":
    show_stock_list()
elif st.session_state.page == "Stock Data":
    show_stock_data()
elif st.session_state.page == "Forecast":
    show_forecast()
