import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\qts\strategies\reliance_fetched_data_1year_15m.csv")
# df = df[:200]
df.rename(columns={'open':'Open','high':'High','low':"Low",'close':'Close','volume':'Volume'},inplace=True)

class MomentumStrategy(Strategy):
    small_threshold = 0
    large_threshold = 3
    
    def momentum(self, data):
        return data.pct_change(periods=7).to_numpy() * 100

    def init(self):
        self._pct_change_Long = resample_apply("2h", self.momentum, self.data.Close)
        self.pct_change_Short = resample_apply("30T", self.momentum, self.data.Close)

    def next(self):
        change_long = self._pct_change_Long[-1]
        change_short = self.pct_change_Short[-1]

        if self.position:
            if self.position.is_long and change_short < self.small_threshold:
                self.position.close()
            elif self.position.is_short and change_short > -1 * self.small_threshold:
                self.position.close()
        else:
            if change_long > self.large_threshold and change_short > self.small_threshold:
                self.buy()
            elif change_long < -1 * self.large_threshold and change_short < -1 * self.small_threshold:
                self.sell()

if __name__ == "__main__":
    bt = Backtest(df, MomentumStrategy, cash=10000, commission=0)
    stats = bt.optimize(
        small_threshold=list(np.arange(0, 1.1, 0.2)),
        large_threshold=list(np.arange(4, 5.1, 0.2)),
        max_tries=50,
        n_jobs=1,
        maximize="Equity Final [$]"
    )
    print(stats)
    bt.plot()
