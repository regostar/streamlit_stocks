# app.py

from stock_data import show_stock_data
from forecast import show_forecast
import streamlit as st
from streamlit_option_menu import option_menu

# Sidebar menu for navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Stock Data", "Forecast"],
                           icons=["bar-chart-line", "graph-up"],
                           menu_icon="menu-app-fill", default_index=0)

# Display the appropriate page based on user selection
if selected == "Stock Data":
    show_stock_data()
elif selected == "Forecast":
    show_forecast()
