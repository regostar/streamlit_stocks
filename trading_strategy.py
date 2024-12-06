import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(200, return_sequences=True))(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(150, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    
    output = Dense(3, activation='softmax')(x)  # Buy/Sell/Hold as output classes
    model = Model(inputs=input_layer, outputs=output)

    optimizer = Adam(learning_rate=2e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_data(stock_data, n_steps=60):
    """Prepare sequences for LSTM and create labels."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    
    X, y = [], []

    for i in range(n_steps, len(scaled_data) - 1):
        X.append(scaled_data[i - n_steps:i, 0])

        # Label for Buy, Hold, Sell - Here we define a simple strategy
        if scaled_data[i + 1, 0] > scaled_data[i, 0]:
            y.append([1, 0, 0])  # Buy
        elif scaled_data[i + 1, 0] == scaled_data[i, 0]:
            y.append([0, 1, 0])  # Hold
        else:
            y.append([0, 0, 1])  # Sell

    return np.array(X), np.array(y), scaler

def backtest_strategy(stock_data, model, scaler, n_steps=60):
    """Simulate a portfolio using buy/sell/hold strategy."""
    # Prepare the backtesting dataset
    scaled_data = scaler.transform(stock_data['Close'].values.reshape(-1, 1))
    X = []

    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i, 0])
    
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Initialize portfolio
    portfolio_value = 10000  # Starting with $10,000
    cash = portfolio_value
    shares = 0

    # Backtesting loop
    for i in range(len(X)):
        prediction = model.predict(X[i].reshape(1, X.shape[1], 1))
        action = np.argmax(prediction)  # 0 = Buy, 1 = Hold, 2 = Sell

        current_price = stock_data['Close'].iloc[i + n_steps]

        if action == 0 and cash > 0:  # Buy
            shares = cash / current_price
            cash = 0
        elif action == 2 and shares > 0:  # Sell
            cash = shares * current_price
            shares = 0

    print("Backtesting done")
    # Final portfolio value
    final_portfolio_value = cash if shares == 0 else (shares * stock_data['Close'].iloc[-1])
    print("Final portfolio value", final_portfolio_value)
    
    return final_portfolio_value

def show_trading_strategy():
    st.title("Stock Buy/Sell/Hold Strategy Using LSTM")

    # Sidebar for stock selection and date range
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # Retrieve stock data using yfinance
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if stock data is empty or invalid
        if stock_data is None or stock_data.empty:
            st.error("No data found for the given ticker or invalid date range.")
            return

        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(stock_data.tail())

        # Prepare data
        n_steps = 60
        X, y, scaler = prepare_data(stock_data, n_steps)

        # Create the LSTM model
        model = create_lstm_model(input_shape=(n_steps, 1))

        # Display progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train the model and update the progress bar
        epochs = 10
        for epoch in range(epochs):
            model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=1, batch_size=32, verbose=0)
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Training model... Epoch {epoch + 1}/{epochs}")

        status_text.text("Model training complete!")

        # Backtest the strategy with the trained model
        final_value = backtest_strategy(stock_data, model, scaler, n_steps)

        # Display the results
        st.subheader("Backtest Results")
        st.write(f"Final Portfolio Value: ${final_value:,.2f}")

        # Plot the results
        st.subheader("Portfolio Value Over Time")
        st.line_chart(stock_data['Close'])  # You can plot the final portfolio value over time here

    except Exception as e:
        st.error(f"An error occurred: {e}")
