# Stock-Price-Prediction---5-future-days
Absolutely not accurate (at all), made it just for fun. 80% ChatGPT, 15% Google, 5% my tiny brain.

Download these packages to run the code. 
pip install yfinance pandas numpy scikit-learn tensorflow plotly ta

These are the 6 indicators I used to analyse the past 60 days' movement of the stock, which is fed to the LSTM model,
and predict the upcoming 5-day stock price movement by learning the pattern:

1. Simple Moving Average (20-day SMA)
2. Relative Strength Index (14-day RSI)
3. MACD difference (MACD line minus signal line)
4. Bollinger Band High and Bollinger Band Low
5. Close price (lagged sequences)
