import pandas as pd
import yfinance as yf

# Fetch historical stock data
msft = yf.download("MSFT", start="2020-01-01", end="2025-01-01")

# Simple Moving Average
def calculate_sma(prices, window):
    return prices.rolling(window=window).mean()

msft['SMA_50'] = calculate_sma(msft['Close'], 50)

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = prices.copy()  # Initialize EMA with prices
    for i in range(1, len(prices)):
        ema.iloc[i] = (prices.iloc[i] * alpha) + (ema.iloc[i-1] * (1 - alpha))
    return ema

msft['EMA_20'] = calculate_ema(msft['Close'], 20)

def calculate_rsi(prices, window):
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

msft['RSI'] = calculate_rsi(msft['Close'], 14)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

msft['MACD'], msft['MACD_Signal'], msft['MACD_Hist'] = calculate_macd(msft['Close'])

def calculate_bollinger_bands(prices, window, num_std_dev):
    sma = calculate_sma(prices, window)
    rolling_std = prices.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return sma, upper_band, lower_band

msft['BB_Middle'], msft['BB_Upper'], msft['BB_Lower'] = calculate_bollinger_bands(msft['Close'], 20, 2)

import matplotlib.pyplot as plt

# Plotting the stock price with SMA and EMA
plt.figure(figsize=(14, 7))
plt.plot(msft['Close'], label='Close Price', alpha=0.8)
plt.plot(msft['SMA_50'], label='50-day SMA', linestyle='--', alpha=0.8)
plt.plot(msft['EMA_20'], label='20-day EMA', linestyle='--', alpha=0.8)
plt.title("Microsoft Stock Price with Moving Averages")
plt.legend()
plt.show()

# Plot RSI
plt.figure(figsize=(14, 5))
plt.plot(msft['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='green', linestyle='--', label='Oversold')
plt.title("Relative Strength Index (RSI)")
plt.legend()
plt.show()
