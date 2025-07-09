# Prompt user for ticker
ticker = input("Enter a stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
if not ticker.isalpha():
    raise ValueError("Invalid ticker symbol. Please enter only alphabetic characters.")

import yfinance as yf
import pandas as pd
import datetime
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.callbacks import EarlyStopping

# Set end date to today
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Download stock data  
stock_data = yf.download(ticker, start='2019-10-01', end=end_date, auto_adjust=False)

# Flatten MultiIndex columns if present
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

# Debug info
print(f"\nColumns after flattening: {stock_data.columns.tolist()}")

# Ensure 'Close' column exists
if 'Close' not in stock_data.columns:
    raise KeyError("No 'Close' column found in downloaded stock data.")

# Ensure 'Close' is float
stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data.dropna(subset=['Close'], inplace=True)


# === Calculate Technical Indicators ===
sma = ta.trend.SMAIndicator(stock_data['Close'], window=20).sma_indicator()
rsi = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()
macd = ta.trend.MACD(stock_data['Close']).macd_diff()

bb = ta.volatility.BollingerBands(stock_data['Close'])
bollinger_h = bb.bollinger_hband()
bollinger_l = bb.bollinger_lband()

# Assign indicators (flatten if needed)
stock_data['SMA'] = sma.values.flatten()
stock_data['RSI'] = rsi.values.flatten()
stock_data['MACD'] = macd.values.flatten()
stock_data['Bollinger_H'] = bollinger_h.values.flatten()
stock_data['Bollinger_L'] = bollinger_l.values.flatten()

# Drop rows with NaN values introduced by indicators
stock_data.dropna(inplace=True)

# === Select features ===
features = ['Close', 'SMA', 'RSI', 'MACD', 'Bollinger_H', 'Bollinger_L']
data = stock_data[features].values

# === Normalize data ===
features_scaler = MinMaxScaler()
scaled_data = features_scaler.fit_transform(data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(stock_data[['Close']].values)

# Split the data into training (80%) and testing (20%) sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]


def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])  
        y.append(data[i, 0])            
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Initialize the model
model = Sequential()

# Add LSTM layers
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], len(features)))
model.add(LSTM(units=100, return_sequences=True, input_shape=(60, len(features))))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(
    monitor='loss',         # you could also use 'val_loss' if using validation_split
    patience=3,             # stops after 3 epochs with no improvement
    restore_best_weights=True
)

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, callbacks=[early_stop])

# === Predict on test data ===
predictions = model.predict(x_test)
predictions = close_scaler.inverse_transform(predictions)

# Inverse transform actual test data
y_test_scaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# === Predict the next 5 trading days ===
future_days = 5
future_predictions = []
last_sequence = scaled_data[-60:].copy()

for _ in range(future_days):
    input_seq = np.reshape(last_sequence, (1, 60, len(features)))
    predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
    future_predictions.append(predicted_scaled)

    # Create next feature row by copying the last one and replacing Close value
    next_features = last_sequence[-1].copy()
    next_features[0] = predicted_scaled  # Replace only the 'Close' value

    # Append it while maintaining shape (60, 6)
    last_sequence = np.concatenate([last_sequence[1:], [next_features]], axis=0)

# Inverse transform future predictions
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_prices = close_scaler.inverse_transform(future_predictions)

# === Metrics ===
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_scaled, predictions)
rmse = np.sqrt(mse)

print(f"\n✅ Prediction complete for: {ticker}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.4f}")

print("\n📅 Predicted Close Prices for the Next 5 Trading Days:")
for i, price in enumerate(future_prices, start=1):
    print(f"Day {i}: ${price[0]:.2f}")

# === Plotting everything ===
import plotly.graph_objs as go
from pandas.tseries.offsets import BDay
fig = go.Figure()

# Plot actual test prices
fig.add_trace(go.Scatter(
    x=stock_data.index[-len(y_test):],
    y=y_test_scaled.flatten(),
    mode='lines',
    name='Actual Price'
))

# Plot predicted test prices
fig.add_trace(go.Scatter(
    x=stock_data.index[-len(y_test):],
    y=predictions.flatten(),
    mode='lines',
    name='Predicted Price'
))

# Plot next 5-day future predictions
last_date = stock_data.index[-1]
future_dates = [last_date + BDay(i) for i in range(1, future_days + 1)]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_prices.flatten(),
    mode='lines+markers',
    name='Next 5-Day Forecast',
    line=dict(dash='dot')
))

fig.update_layout(
    title=f'{ticker} Stock Price Prediction',
    xaxis_title='Date',
    yaxis_title='Stock Price (USD)',
    legend_title='Legend'
)

fig.show()
