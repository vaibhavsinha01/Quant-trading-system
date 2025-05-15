import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
# import talib
from ta.trend import ADXIndicator,EMAIndicator

def dema(series, period):
    """Calculate Double Exponential Moving Average"""
    # ema1 = talib.EMA(series, timeperiod=period)
    # ema2 = talib.EMA(ema1, timeperiod=period)
    ema1 = EMAIndicator(close=series,window=period).ema_indicator()
    ema2 = EMAIndicator(close=ema1,window=period).ema_indicator()
    return 2 * ema1 - ema2

def adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # return talib.ADX(high, low, close, timeperiod=period)
    return ADXIndicator(high=high,low=low,close=close,window=period).adx()

def squeeze_momentum(df, length=20, mult=2.0, lengthKC=20, multKC=1.5, use_truerange=True):
    df = df.copy()
    
    # Bollinger Bands
    df['basis'] = df['Close'].rolling(window=length).mean()
    df['dev'] = mult * df['Close'].rolling(window=length).std()
    df['upperBB'] = df['basis'] + df['dev']
    df['lowerBB'] = df['basis'] - df['dev']
    
    # Keltner Channels
    df['ma'] = df['Close'].rolling(window=lengthKC).mean()
    if use_truerange:
        df['tr'] = np.maximum(df['High'] - df['Low'],
                np.maximum(abs(df['High'] - df['Close'].shift(1)),
                            abs(df['Low'] - df['Close'].shift(1))))
    else:
        df['tr'] = df['High'] - df['Low']
        
    df['rangema'] = df['tr'].rolling(window=lengthKC).mean()
    df['upperKC'] = df['ma'] + df['rangema'] * multKC
    df['lowerKC'] = df['ma'] - df['rangema'] * multKC
    
    # Squeeze conditions
    df['sqzOn'] = (df['lowerBB'] > df['lowerKC']) & (df['upperBB'] < df['upperKC'])
    df['sqzOff'] = (df['lowerBB'] < df['lowerKC']) & (df['upperBB'] > df['upperKC'])
    df['noSqz'] = ~(df['sqzOn'] | df['sqzOff'])
    
    # Squeeze momentum value
    mid_high_low = (df['High'].rolling(lengthKC).max() + df['Low'].rolling(lengthKC).min()) / 2
    val_input = df['Close'] - ((mid_high_low + df['Close'].rolling(lengthKC).mean()) / 2)
    df['val'] = val_input.rolling(window=lengthKC).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] * (len(x) - 1) + np.polyfit(range(len(x)), x, 1)[1],
        raw=False
    )
    
    # Bar color logic
    df['val_prev'] = df['val'].shift(1)
    df['bcolor'] = np.where(df['val'] > 0,
                            np.where(df['val'] > df['val_prev'], 'lime', 'green'),
                            np.where(df['val'] < df['val_prev'], 'red', 'maroon'))
    
    df['scolor'] = np.where(df['noSqz'], 'blue',
                            np.where(df['sqzOn'], 'black', 'gray'))
    
    return df['val'], df['bcolor'], df['scolor'], df['sqzOn'], df['sqzOff'], df['noSqz']

class SqueezeStrategy(Strategy):
    # Strategy parameters
    length = 20
    mult = 2.0
    lengthKC = 20
    multKC = 1.5
    dema_period = 50
    adx_period = 14
    
    def init(self):
        # Calculate DEMA
        self.dema50 = self.I(dema, self.data.Close, self.dema_period)
        
        # Calculate ADX
        self.adx_indicator = self.I(adx, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.adx_slope = self.I(lambda x: pd.Series(x).diff(), self.adx_indicator)
        
        # Get indicators from squeeze_momentum
        df = pd.DataFrame({
            'Open': self.data.Open, 
            'High': self.data.High, 
            'Low': self.data.Low, 
            'Close': self.data.Close
        })
        
        val, bcolor, scolor, sqzOn, sqzOff, noSqz = squeeze_momentum(
            df, self.length, self.mult, self.lengthKC, self.multKC
        )
        
        self.squeeze_val = self.I(lambda x: x, val)
        self.squeeze_bcolor = self.I(lambda x: x, bcolor)
    
    def next(self):
        # Check for long entry conditions
        long_condition = (
            self.adx_slope[-1] > 0 and  # ADX slope is positive
            self.dema50[-1] > self.data.Close[-1] and  # DEMA50 above price
            self.squeeze_val[-1] > 0 and  # Squeeze momentum positive
            self.squeeze_bcolor[-1] == 'lime'  # Light green bar (increasing positive momentum)
        )
        
        # Check for short entry conditions
        short_condition = (
            self.adx_slope[-1] < 0 and  # ADX slope is negative
            self.dema50[-1] < self.data.Close[-1] and  # DEMA50 below price
            self.squeeze_val[-1] < 0 and  # Squeeze momentum negative
            self.squeeze_bcolor[-1] == 'red'  # Light red bar (increasing negative momentum)
        )
        
        # Simple trading logic
        if not self.position:
            # If no position, check entry conditions
            if long_condition:
                self.buy()
            elif short_condition:
                self.sell()
        else:
            # If in a long position, check if we should exit
            if self.position.is_long and (
                self.adx_slope[-1] < 0 or 
                self.dema50[-1] < self.data.Close[-1] or 
                self.squeeze_val[-1] < 0
            ):
                self.position.close()
            
            # If in a short position, check if we should exit
            elif self.position.is_short and (
                self.adx_slope[-1] > 0 or 
                self.dema50[-1] > self.data.Close[-1] or 
                self.squeeze_val[-1] > 0
            ):
                self.position.close()

# Main function to run the backtest
def run_backtest(data, initial_cash=10000, commission=0.002):
    bt = Backtest(data, SqueezeStrategy, cash=initial_cash, commission=commission)
    stats = bt.run()
    return bt, stats

# Example usage
if __name__ == "__main__":
    # Load your data
    # You need OHLC data in a pandas DataFrame
    # Example: df = pd.read_csv('your_data.csv')
    
    # Let's assume we have sample data
    import yfinance as yf
    data = yf.download('SPY', start='2020-01-01', end='2023-12-31')
    
    # Run backtest
    bt, stats = run_backtest(data)
    
    # Print stats
    print(stats)
    
    # Plot the equity curve and trades
    bt.plot()