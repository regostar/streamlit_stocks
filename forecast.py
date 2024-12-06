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
            # Add title with question mark icon for Stock Price Forecast with no space below it
            st.markdown(f"<h2 style='margin-bottom: 0px;'>Stock Price Forecast for {ticker}</h2>", unsafe_allow_html=True)
            st.markdown(
                """
                <span style="cursor: pointer; font-size: 20px;" title="The graph displays historical stock price movement (solid blue line) along with a predicted future trend (light blue line), showing an overall upward trajectory with some fluctuations. The forecast suggests a continued gradual increase in the stock's value.">❓</span>
                """,
                unsafe_allow_html=True
            )

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

            st.markdown(
                """
                <h3 style="display: inline;">Forecast Components</h3>
                <span style="cursor: pointer; font-size: 20px;" title="The first graph shows the overall price trend over time with future predictions, while the second graph displays typical price patterns across different weekdays. The third graph illustrates how prices fluctuate throughout trading hours, with peaks during market hours and dips during off-hours.">❓</span>
                """, unsafe_allow_html=True)
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            st.markdown(
                """
                <h3 style="display: inline;">Price Line vs Forecast Trend Line</h3>
                <span style="cursor: pointer; font-size: 20px;" title="The graph compares actual historical stock prices (teal line) against predicted future values (red line), with blue dotted lines showing potential upper and lower price ranges. The forecast components break down the stock's behavior patterns across weekly trading days and daily trading hours.">❓</span>
                """, unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Price line', line=dict(color='teal')))
            fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast trend line', line=dict(color='red')))
            fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot', color='blue')))
            fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot', color='blue')))
            fig3.update_layout(xaxis_title="Date", yaxis_title="Price", margin={"t": 10, "b": 10})  
            st.plotly_chart(fig3)

            st.markdown(
                """
                <h3 style="display: inline;">Trend Component</h3>
                <span style="cursor: pointer; font-size: 20px;" title="The trend component graph shows the overall long-term direction of the stock price, removing daily and weekly fluctuations.">❓</span>
                """, unsafe_allow_html=True)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=future['ds'], y=forecast['trend'], mode='lines', name="Trend"))
            fig4.update_layout(xaxis_title="Date", yaxis_title="Trend Effect", margin={"t": 10, "b": 10})  
            st.plotly_chart(fig4)

            st.markdown(
                """
                <h3 style="display: inline;">Weekly Component</h3>
                <span style="cursor: pointer; font-size: 20px;" title="The weekly component graph shows how stock prices typically fluctuate across different days of the week, displayed as vertical bars that represent whether prices tend to be higher or lower on specific weekdays. These recurring weekly patterns help understand regular price movements that happen during a typical trading week.">❓</span>
                """, unsafe_allow_html=True)
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=future['ds'], y=forecast['weekly'], mode='lines', name="Weekly"))
            fig5.update_layout(xaxis_title="Date", yaxis_title="Weekly Trend Effect", margin={"t": 10, "b": 10})  
            st.plotly_chart(fig5)

        else:
            st.error("No data found. Please check the ticker.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
