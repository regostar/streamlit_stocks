import streamlit as st
import pandas as pd
import yfinance as yf

@st.cache_data
def load_stock_list():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    stock_list = pd.read_csv(url)
    stock_list.rename(columns={"Symbol": "Symbol", "Security": "Name", "GICS Sector": "Industry"}, inplace=True)
    stock_list = stock_list[['Symbol', 'Name', 'Industry']]
    return stock_list

def fetch_stock_data(symbols):
    prices, market_caps, pe_ratios = [], [], []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            history_data = ticker.history(period="1d")
            price = history_data['Close'].iloc[-1] if not history_data.empty else 'N/A'
            info = ticker.info
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            prices.append(price)
            market_caps.append(market_cap)
            pe_ratios.append(pe_ratio)
        except Exception:
            prices.append('N/A')
            market_caps.append('N/A')
            pe_ratios.append('N/A')
    return prices, market_caps, pe_ratios

def show_stock_list():
    st.title("Stock List")
    st.write("Browse and filter the list of stocks with current data.")

    # Load the full stock list
    stocks_data = load_stock_list()
    search_term = st.text_input("Search by Symbol or Name", "")
    industry_filter = st.selectbox("Filter by Industry", options=["All"] + stocks_data['Industry'].unique().tolist())

    # Apply search and filter criteria
    filtered_data = stocks_data[
        (stocks_data['Symbol'].str.contains(search_term, case=False)) | 
        (stocks_data['Name'].str.contains(search_term, case=False))
    ]
    if industry_filter != "All":
        filtered_data = filtered_data[filtered_data['Industry'] == industry_filter]

    # Pagination setup
    items_per_page = 10
    total_items = len(filtered_data)
    total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)
    page_number = st.number_input("Page Number", min_value=1, max_value=total_pages, step=1)
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_data = filtered_data.iloc[start_idx:end_idx]

    # Fetch data for only the visible stocks on the current page
    symbols = paginated_data['Symbol'].tolist()
    prices, market_caps, pe_ratios = fetch_stock_data(symbols)

    # Add fetched data to the paginated DataFrame
    paginated_data = paginated_data.copy()
    paginated_data['Price'] = prices
    paginated_data['Market Cap'] = market_caps
    paginated_data['P/E Ratio'] = pe_ratios

    # Display table with buttons in the last column
    for idx, row in paginated_data.iterrows():
        cols = st.columns(len(row) + 1)  # Extra column for button
        for i, value in enumerate(row):
            cols[i].write(value)
        # Add button in the last column
        if cols[len(row)].button("View Details", key=row['Symbol']):
            st.session_state.selected_stock = row['Symbol']
            st.session_state.page = "Stock Data"  # Navigate to Stock Data page

    st.write(f"Showing page {page_number} of {total_pages}")
