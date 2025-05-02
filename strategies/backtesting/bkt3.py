import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# --- Indicators ---

def harris_signal(df):
    signal = np.zeros(len(df))
    for i in range(3, len(df)):
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
            df['Low'].iloc[i] > df['High'].iloc[i-2] and 
            df['High'].iloc[i-2] > df['Low'].iloc[i-1]):
            signal[i] = 1
        elif (df['High'].iloc[i] < df['High'].iloc[i-1] and 
              df['Low'].iloc[i] < df['High'].iloc[i-2] and 
              df['High'].iloc[i-2] < df['Low'].iloc[i-1]):
            signal[i] = -1
    return signal

def t3_ema(df, length=8, vfactor=0.7, price_col='Close'):
    e1 = df[price_col].ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    e4 = e3.ewm(span=length, adjust=False).mean()
    e5 = e4.ewm(span=length, adjust=False).mean()
    e6 = e5.ewm(span=length, adjust=False).mean()

    c1 = -vfactor**3
    c2 = 3 * vfactor**2 + 3 * vfactor**3
    c3 = -6 * vfactor**2 - 3 * vfactor - 3 * vfactor**3
    c4 = 1 + 3 * vfactor + vfactor**3 + 3 * vfactor**2

    return c1*e6 + c2*e5 + c3*e4 + c4*e3

# --- Strategy ---

class HarrisStrategy(Strategy):
    t3_fast_len = 8
    t3_slow_len = 13
    vfactor_int = 700      # vfactor * 1000
    tp_pct_int = 5         # tp_pct * 1000
    sl_pct_int = 3         # sl_pct * 1000

    def init(self):
        vfactor = self.vfactor_int / 1000
        self.tp_pct = self.tp_pct_int / 1000
        self.sl_pct = self.sl_pct_int / 1000

        self.t3_fast = self.I(t3_ema, self.data.df, self.t3_fast_len, vfactor)
        self.t3_slow = self.I(t3_ema, self.data.df, self.t3_slow_len, vfactor)
        self.h_signal = self.I(harris_signal, self.data.df)

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[-1]

        t3_fast_slope = self.t3_fast[-1] - self.t3_fast[-2]
        t3_slow_slope = self.t3_slow[-1] - self.t3_slow[-2]

        if not self.position:
            if self.h_signal[i] == 1 and (t3_fast_slope > 0 or t3_slow_slope > 0):
                self.buy(
                    sl=price * (1 - self.sl_pct),
                    tp=price * (1 + self.tp_pct)
                )
            elif self.h_signal[i] == -1 and (t3_fast_slope < 0 or t3_slow_slope < 0):
                self.sell(
                    sl=price * (1 + self.sl_pct),
                    tp=price * (1 - self.tp_pct)
                )

# --- Load Data ---

df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\qts\features.csv")
df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)

# --- Run Backtest ---

bt = Backtest(df, HarrisStrategy, cash=100000)

stats = bt.run()
print(stats)
bt.plot()

# # --- Wrap Optimization inside main guard ---
