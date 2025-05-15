import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import GOOG, EURUSD

class WilliamsRStrategy(Strategy):
    def init(self):
        self.period = 14

        # We need to calculate highest_high and lowest_low rolling externally on pd.Series
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        highest_high = high.rolling(self.period).max()
        lowest_low = low.rolling(self.period).min()

        # Calculate Williams %R using self.I to register it for backtesting
        self.willr = self.I(
            lambda hh, ll, c: -100 * (hh - c) / (hh - ll),
            highest_high, lowest_low, close
        )

        # Previous values, shifted by 1
        self.prev_high = self.I(lambda x: x.shift(1), high)
        self.willr_prev = self.I(lambda x: x.shift(1), pd.Series(self.willr))

    def next(self):
        i = len(self.data) - 1

        if not self.position:
            if self.willr_prev[i] > -90 and self.willr[i] <= -90:
                self.buy()
        else:
            if (self.willr_prev[i] < -30 and self.willr[i] >= -30) or self.data.Close[i] > self.prev_high[i]:
                self.position.close()

def run_backtest(df, asset_name):
    print(f"\n📊 Backtest Report for {asset_name}")
    bt = Backtest(df, WilliamsRStrategy, cash=100000, commission=0, exclusive_orders=True)
    results = bt.run()
    print(results)
    bt.plot()

run_backtest(GOOG, "GOOG")
run_backtest(EURUSD, "EURUSD")
