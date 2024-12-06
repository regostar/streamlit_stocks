import streamlit as st
from streamlit_option_menu import option_menu
from stock_data import show_stock_data
from forecast import show_forecast

from stock_list import show_stock_list  # Import the new page
from chatbot import show_chatbot
from trading_strategy import show_trading_strategy

# Sidebar menu for navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Stock Data", "Forecast", "Stock List", "Chatbot", "Trading Strategy"],
                           icons=["bar-chart-line", "graph-up", "list", "chat"],
                           menu_icon="menu-app-fill", default_index=0)

# Display the appropriate page based on user selection
if selected == "Stock Data":
    show_stock_data()
elif selected == "Forecast":
    show_forecast()
elif selected == "Stock List":
    show_stock_list()  # Show the stock list page
# elif selected == "Chatbot":
#     show_chatbot()  # Show chatbot page
elif selected == "Trading Strategy":
    show_trading_strategy()  # Show the trading strategy page
elif selected == "Chatbot":
    show_chatbot()
