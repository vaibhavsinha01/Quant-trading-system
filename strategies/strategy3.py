# this is harris's strategy used for getting 74% win-rate this is a long only strategy

import numpy as np
import pandas as pd
from SmartApi import SmartConnect
from datetime import datetime,timedelta
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
            else:
                print(f"Data fetched successfully: {self.df.tail()}")

        except Exception as e:
            print(f"Error in fetch_data: {e}")

    def feature_engineering(self):
        # self.df = self.volatI.directional_volatility(self.df)
        self.candlI.calculate_HeikenAshi_indicators(self.df)
        self.df['t3_slow'],self.df['t3_fast'] = self.trendI.t3ribbon(self.df)
        self.df['harris_signal'] = self.candlI.harris_signal(self.df)
        print(self.df)
        self.df.to_csv("features.csv")

    def generate_signals(self):
        for i in range(len(self.df)):
            if(self.df.loc[i,'harris_signal'] == 1 and (self.df.loc[i,'t3_slow']>=self.df.loc[i-1,'t3_slow'] or self.df.loc[i,'t3_fast']>=self.df.loc[i-1,'t3_fast'])):
                self.df.loc[i,'signal'] = 1

            elif(self.df.loc[i,'harris_signal'] == -1 and (self.df.loc[i,'t3_slow']<=self.df.loc[i-1,'t3_slow'] or self.df.loc[i,'t3_fast']<=self.df.loc[i-1,'t3_fast'])):
                self.df.loc[i,'signal'] = -1

            else:
                self.df.loc[i,'signal'] = 0
        
        self.df.dropna(axis=0,inplace=True)

    def execute_signals(self): # the entire logic tm/rm/pm would be used here this is a simple application
        if self.df['signal'].iloc[-1] == 1:
            print("A buy signal will be placed")
        elif self.df['signal'].iloc[-1] == -1:
            print("A sell signal will be placed")
        else:
            print("No trade would be currently placed")

    def run(self):
        while True:
            self.fetch_data()
            self.feature_engineering()
            self.generate_signals()
            self.execute_signals()
            time.sleep(20)

if __name__ == "__main__":
    s1 = Strategy1()
    s1.run()
