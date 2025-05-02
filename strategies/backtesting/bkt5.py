# trading using autobnn system

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autobnn.autobnn import AutoBNN
import matplotlib.pyplot as plt

# Load your OHLCV DataFrame
# Ensure it has ['open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\qts\features.csv')

# Drop NaNs and sort
df = df.sort_index()
df.dropna(inplace=True)

# === Feature Engineering ===
# EMA200
df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()

# RSI
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

# Label: 5-candle future return
lookahead = 5
df['future_return'] = df['close'].pct_change(lookahead).shift(-lookahead)

# Binary classification: will price increase or not
df['target'] = (df['future_return'] > 0).astype(int)

# Drop rows with NaN after indicator and return calculations
df.dropna(inplace=True)

# Features and Target
features = ['ema200', 'rsi']
X = df[features].values
y = df['target'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === AutoBNN model ===
model = AutoBNN(mode='classification', n_iter=50, verbose=True)
model.fit(X_train, y_train)

# === Evaluate ===
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# === Predict future ===
latest_data = scaler.transform(df[features].iloc[-1:].values)
prediction = model.predict(latest_data)
print(f"Prediction (1=UP, 0=DOWN): {prediction[0]}")

# === Optional: Plot RSI and EMA200 ===
plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close Price', alpha=0.6)
plt.plot(df['ema200'], label='EMA200', color='red')
plt.title("Close Price with EMA200")
plt.legend()
plt.show()
