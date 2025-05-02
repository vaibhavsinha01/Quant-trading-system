import numpy as np
import pandas as pd
from SmartApi import SmartConnect
from datetime import datetime, timedelta
from indicators.volatility import VolatilityIndicator
from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicator
from indicators.volume import VolumeIndicator
from indicators.candles import CandleIndicators
from broker.angel import AngelBrokerWrapper
import creds 
import time

class Strategy1:
    def __init__(self):
        self.api_key = creds.api_key
        self.username = creds.username
        self.password = creds.password
        self.token = creds.token
        self.broker = AngelBrokerWrapper(api_key=self.api_key, username=self.username, password=self.password, token=self.token)
        self.trendI = TrendIndicators()
        self.momenI = MomentumIndicator()
        self.volatI = VolatilityIndicator()
        self.volumI = VolumeIndicator()
        self.candlI = CandleIndicators()

    def fetch_data(self):
        try:
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            start_time = market_open_time - timedelta(days=creds.use_data_for_x_days)
            self.df = self.broker.get_candle_data(
                creds.exchange,
                creds.token_id,
                creds.timeframe,
                start_time.strftime("%Y-%m-%d %H:%M"),
                current_time.strftime("%Y-%m-%d %H:%M")
            )
            print("The data is fetched and the current df is \n")
            print(self.df)
            print("\n")
            # Ensure renaming is assigned back to self.df
            self.df = self.df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            if self.df is None or self.df.empty:
                print("Fetched data is empty. Check data source.")
                return False
            else:
                print(f"Data fetched successfully: {self.df.tail()}")
                self.df.to_csv('features.csv')
                exit(0)
                return True

        except Exception as e:
            print(f"Error in fetch_data: {e}")
            return False
        

    def mfi(self, df, length=14):
        """
        Money Flow Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for MFI calculation
            
        Returns:
        --------
        pandas.Series: MFI values
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * df['volume']
        
        # Get money flow direction
        diff = typical_price.diff()
        
        # Create positive and negative money flow series
        positive_flow = pd.Series(np.where(diff > 0, raw_money_flow, 0), index=df.index)
        negative_flow = pd.Series(np.where(diff < 0, raw_money_flow, 0), index=df.index)
        
        # Sum positive and negative money flows over the period
        positive_sum = positive_flow.rolling(window=length).sum()
        negative_sum = negative_flow.rolling(window=length).sum()
        
        # Calculate money flow ratio (handle division by zero)
        money_ratio = np.where(negative_sum != 0, positive_sum / negative_sum, 100)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        return pd.Series(mfi, index=df.index)

    def feature_engineering(self):
        # self.df = self.volatI.directional_volatility(self.df)
        self.candlI.calculate_HeikenAshi_indicators(self.df)
        self.df['atr'] = self.volatI.atr(self.df,window=10)
        self.df['mfi'] = self.mfi(self.df, length=10)  # Use our fixed MFI calculation
        self.df['macd_ema'], self.df['macd_signal'], self.df['macd_hist'] = self.trendI.macd(self.df)
        print(self.df)
        self.df.to_csv("features.csv")

    def generate_signals(self):
        self.df['signal'] = 0
        
        # Start from a safe index to avoid indexing errors
        # Make sure we have enough data before generating signals
        if len(self.df) < 3:
            print("Not enough data to generate signals")
            return
            
        for i in range(2, len(self.df)):
            # Handle potential NaN values
            current_mfi = self.df.loc[i, 'mfi']
            prev_mfi = self.df.loc[i-1, 'mfi']
            prev2_mfi = self.df.loc[i-2, 'mfi']
            
            current_close = self.df.loc[i, 'close']
            prev_close = self.df.loc[i-1, 'close']
            prev2_close = self.df.loc[i-2, 'close']
            
            current_macd_ema = self.df.loc[i, 'macd_ema']
            current_macd_signal = self.df.loc[i, 'macd_signal']
            
            # Skip if any required values are NaN
            if (pd.isna(current_mfi) or pd.isna(prev_mfi) or pd.isna(prev2_mfi) or
                pd.isna(current_close) or pd.isna(prev_close) or pd.isna(prev2_close) or
                pd.isna(current_macd_ema) or pd.isna(current_macd_signal)):
                continue
                
            # Buy signal: MACD line crosses above signal line + MFI divergence
            # if (current_macd_ema >= current_macd_signal and 
            #     (current_mfi >= prev_mfi and current_close <= prev_close) and 
            #     (prev_mfi >= prev2_mfi and prev_close <= prev2_close)):
            #     self.df.loc[i, 'signal'] = 1
            
            # # Sell signal: MACD line crosses below signal line + MFI divergence
            # elif (current_macd_ema <= current_macd_signal and 
            #      (current_mfi <= prev_mfi and current_close >= prev_close) and 
            #      (prev_mfi <= prev2_mfi and prev_close >= prev2_close)):
            #     self.df.loc[i, 'signal'] = -1

            if((self.df.loc[i,'mfi'] <= 40) and (self.df.loc[i,'mfi'] >= self.df.loc[i-1,'mfi'] and self.df.loc[i,'close'] <= self.df.loc[i-1,'close']) and (self.df.loc[i-1,'mfi'] >= self.df.loc[i-2,'mfi'] and self.df.loc[i-1,'close'] <= self.df.loc[i-2,'close'])):
                self.df.loc[i,'signal'] = 1
            elif((self.df.loc[i,'mfi'] >= 60) and (self.df.loc[i,'mfi'] <= self.df.loc[i-1,'mfi'] and self.df.loc[i,'close'] >= self.df.loc[i-1,'close']) and (self.df.loc[i-1,'mfi'] <= self.df.loc[i-2,'mfi'] and self.df.loc[i-1,'close'] >= self.df.loc[i-2,'close'])):
                self.df.loc[i,'signal'] = -1
            # No signal
            else:
                self.df.loc[i, 'signal'] = 0
        
        # Remove rows with NaN values
        self.df = self.df.dropna()

    # def execute_signals(self):
    #     # Check if dataframe has data
    #     if self.df.empty:
    #         print("No data available for signal execution")
    #         return
            
    #     # Check if 'signal' column exists
    #     if 'signal' not in self.df.columns:
    #         print("Signal column does not exist")
    #         return
            
    #     # Get the latest signal safely
    #     try:
    #         latest_signal = self.df['signal'].iloc[-1]
    #         if latest_signal == 1:
    #             print("A buy signal will be placed")
    #         elif latest_signal == -1:
    #             print("A sell signal will be placed")
    #         else:
    #             print("No trade would be currently placed")
    #     except IndexError as e:
    #         print(f"Error accessing signal: {e}")
    #         print(f"DataFrame shape: {self.df.shape}")
    #         print("Please check if DataFrame has data after NaN filtering")

    def run(self):
        while True:
            data_fetched = self.fetch_data()
            if data_fetched:
                self.feature_engineering()
                self.generate_signals()
                # self.execute_signals()
            else:
                print("Skipping strategy execution due to data fetch failure")
            
            print("Waiting for next iteration...")
            time.sleep(20)

if __name__ == "__main__":
    s1 = Strategy1()
    s1.run()

