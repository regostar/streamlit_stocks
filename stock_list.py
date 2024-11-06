import streamlit as st
import pandas as pd
import yfinance as yf

# Fetch S&P 500 stock list from an external CSV file on GitHub
@st.cache_data
def get_stock_list():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    stock_list = pd.read_csv(url)
    
    # Rename columns to standardize naming
    stock_list.rename(columns={"Symbol": "Symbol", "Security": "Name", "GICS Sector": "Industry"}, inplace=True)
    
    # Check if the renaming was successful
    if "Industry" not in stock_list.columns:
        st.error("Error: 'Industry' column not found. Please check the CSV structure.")
        
    return stock_list

# Function to display paginated and searchable stock list
def show_stock_list():
    st.title("Stock List")
    st.write("Browse and filter the list of stocks.")

    # Fetch stock data dynamically
    stocks_data = get_stock_list()

    # Search and filter
    search_term = st.text_input("Search by Symbol or Name", "")
    industry_filter = st.selectbox("Filter by Industry", options=["All"] + stocks_data['Industry'].unique().tolist())

    # Filter stocks based on search and industry selection
    filtered_data = stocks_data[
        (stocks_data['Symbol'].str.contains(search_term, case=False) | 
         stocks_data['Name'].str.contains(search_term, case=False))
    ]
    if industry_filter != "All":
        filtered_data = filtered_data[filtered_data['Industry'] == industry_filter]

    # Pagination
    items_per_page = 5
    page_number = st.number_input("Page Number", min_value=1, max_value=(len(filtered_data) // items_per_page) + 1, step=1)
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_data = filtered_data.iloc[start_idx:end_idx]

    # Display paginated table
    st.write(f"Showing page {page_number}")
    st.table(paginated_data)

# Display the stock list page
show_stock_list()
