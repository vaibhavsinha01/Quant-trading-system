# this file tells about the sentiment of the stock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Stockcheck:
    def __init__(self):
        pass

    def hurst_exponent(self, df, column='close', max_lag=100):
        ts = df[column].values
        lags = range(2, max_lag)
        tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return hurst

    def detect_distribution_days(self,df):
        df = df.copy()
        df['prev_volume'] = df['volume'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['distribution_day'] = (
            (df['close'] < df['prev_close']) &
            (df['volume'] > df['prev_volume']) &
            ((df['prev_close'] - df['close']) / df['prev_close'] > 0.002)
        )
        return df[['date', 'close', 'volume', 'distribution_day']]

    def canslim_sentiment_index(self,C=1, A=1, N=1, S=1, L=1, I=1, M=1):
        """
        | Letter | Factor | Description | Possible Sentiment Signal |
        |--------|--------|-------------|----------------------------|
        | **C** | Current earnings | Quarterly EPS growth |  High growth = Greed |
        | **A** | Annual earnings | Year-over-year EPS growth |  Consistent = Greed |
        | **N** | New product/service/high | Innovation or IPO |  Buzz = Greed |
        | **S** | Supply/demand | Low float + high volume |  Tight supply = Greed |
        | **L** | Leader or laggard | Relative Strength vs peers |  Leader = Greed |
        | **I** | Institutional sponsorship | Mutual fund ownership rising |  Decline = Fear |
        | **M** | Market direction | General trend using indices, distribution days |  Downtrend = Fear |
        """
        # All scores should be between -1 (fear) to +1 (greed)
        factors = [C, A, N, S, L, I, M]
        score = sum(factors) / len(factors)
        
        if score >= 0.7:
            sentiment = "Extreme Greed"
        elif score >= 0.3:
            sentiment = "Greed"
        elif score <= -0.7:
            sentiment = "Extreme Fear"
        elif score <= -0.3:
            sentiment = "Fear"
        else:
            sentiment = "Neutral"
        
        return score, sentiment

if __name__ == "__main__":
    # Simulate price data
    np.random.seed(2)
    dates = pd.date_range(start='2022-01-01', periods=500)
    prices = np.cumsum(np.random.randn(500)) + 100  # Random walk
    volumes = np.random.randint(100000, 500000, size=500)
    df = pd.DataFrame({'date': dates, 'close': prices, 'volume': volumes})

    sc = Stockcheck()

    # Calculate Hurst Exponent
    hurst = sc.hurst_exponent(df)
    print(f"\nHurst Exponent: {hurst:.4f}")
    if hurst < 0.5:
        print("Market is likely mean-reverting (range-bound).")
    elif hurst > 0.5:
        print("Market is likely trending.")
    else:
        print("Market behaves like a random walk.")

    # Detect Distribution Days
    dist_df = sc.detect_distribution_days(df)
    recent_dist_days = dist_df[dist_df['distribution_day']].tail(5)
    print(f"\nRecent Distribution Days:\n{recent_dist_days.to_string(index=False)}")
    score,sentiment = sc.canslim_sentiment_index()
    print(f"\nCANSLIM Sentiment Score: {score:.2f} => {sentiment}")
