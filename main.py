# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
from PIL import Image


# Define function for stock price prediction
def stock_price_prediction(ticker, start_date, end_date, future_days):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Close']]
    stock_data['Days'] = np.arange(len(stock_data))

    X = stock_data[['Days']].values
    y = stock_data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices
    future_X = np.array(range(len(stock_data), len(stock_data) + future_days)).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data['Days'], stock_data['Close'], label="Historical Prices", color="blue")
    ax.plot(range(len(stock_data), len(stock_data) + future_days), future_predictions, color='orange',
            label="Future Predictions")
    ax.set_xlabel("Days")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Stock Price Prediction for {ticker}")
    ax.legend()

    st.pyplot(fig)

    return future_predictions


# Define Streamlit app layout
st.title("Stock Price Prediction App")
st.write("Select a stock, and the app will predict its price for a given number of days into the future.")

# Define available stocks and associated images
stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN"
}

# Sidebar for selecting stock and date inputs
selected_stock = st.sidebar.selectbox("Choose a stock:", list(stocks.keys()))
ticker = stocks[selected_stock]

start_date = st.sidebar.date_input("Select start date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("Select end date", datetime(2023, 1, 1))
future_days = st.sidebar.slider("Number of future days to predict:", min_value=1, max_value=30, value=5)

# Display stock image
image_path = f"images/{ticker}.png"
try:
    image = Image.open(image_path)
    st.image(image, caption=f"{selected_stock}", use_column_width=True)
except FileNotFoundError:
    st.warning(f"No image available for {selected_stock}.")

# Run prediction and display results
if st.sidebar.button("Predict"):
    st.write(f"### Stock Price Prediction for {selected_stock}")
    predicted_prices = stock_price_prediction(ticker, start_date, end_date, future_days)
    st.write("Predicted Future Prices:", predicted_prices)
