# This file decides whether the stock is of trending/reverting nature now.
import numpy as np
import pandas as pd

class Stockcheck:
    def __init__(self):
        pass

    def hurst_exponent(self, df, column='close', max_lag=100):
        ts = df[column].values
        lags = range(2, max_lag)
        tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return hurst

if __name__ == "__main__":
    # Simulate price data
    np.random.seed(2)
    dates = pd.date_range(start='2022-01-01', periods=500)
    prices = np.cumsum(np.random.randn(500)) + 100  # Random walk with noise
    df = pd.DataFrame({'date': dates, 'close': prices})

    # Instantiate the Stockcheck class
    sc = Stockcheck()

    # Calculate the Hurst Exponent
    hurst = sc.hurst_exponent(df)
    print(f"Hurst Exponent: {hurst:.4f}")

    # Interpret result
    if hurst < 0.5:
        print("Market is likely mean-reverting (e.g., range-bound).")
    elif hurst > 0.5:
        print("Market is likely trending.")
    else:
        print("Market is likely behaving as a random walk.")
