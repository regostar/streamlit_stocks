import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import datetime
import plotly.graph_objs as go

def show_forecast():
    st.title("Stock Price Forecasting")
    st.write("Forecast future stock prices for up to 1 year.")

    # Sidebar inputs for stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", "AAPL")
    period = st.sidebar.slider("Forecast Period (Months)", 1, 12, 6)
    
    # Download historical stock data
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * 2)  # 2 years of data for better forecasting
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            st.subheader(f"Stock Price Forecast for {ticker}")

            # Prepare data for Prophet
            df = stock_data.reset_index()[['Date', 'Close']]
            df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone information
            df.columns = ['ds', 'y']

            # Train Prophet model
            model = Prophet(daily_seasonality=True)
            model.fit(df)

            # Create future dataframe for forecasting
            future = model.make_future_dataframe(periods=period * 30)  # Monthly periods
            forecast = model.predict(future)

            # Custom Plotly chart for forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))

            st.plotly_chart(fig)

            # Display forecast components (trend, seasonality) using Prophet's plot components
            st.write("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            # Price Line vs Forecast Trend Line
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Price line', line=dict(color='teal')))
            fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast trend line', line=dict(color='red')))
            fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot', color='blue')))
            fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot', color='blue')))
            fig3.update_layout(title="Price Line vs Forecast Trend Line", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig3)

            # Interactive Forecast Components
            components = ['trend', 'weekly', 'yearly']
            for component in components:
                if component in forecast.columns:
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(x=future['ds'], y=forecast[component], mode='lines', name=component))
                    fig4.update_layout(title=f"{component.capitalize()} Component", xaxis_title="Date", yaxis_title="Weekly Trend Effect")
                    st.plotly_chart(fig4)


        else:
            st.error("No data found. Please check the ticker.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
