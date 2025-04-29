import numpy as np
import pandas as pd
from SmartApi import SmartConnect
from datetime import datetime,timedelta
from indicators.volatility import VolatilityIndicator
from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicator
from indicators.volume import VolumeIndicator
from broker.angel import AngelBrokerWrapper
import creds 

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
        self.df = self.volatI.directional_volatility(self.df)
        self.df['cci'] = self.momenI.cci(self.df)
        self.df['tsv'] = self.volumI.tsv(self.df)
        self.df['macd_ema'],self.df['macd_signal'],self.df['histogram'] = self.trendI.macd(self.df)
        print(self.df)

    def generate_signals(self):
        for i in range(len(self.df)):
            if(self.df.loc[i,'macd_ema']>self.df.loc[i,'macd_signal'] and self.df.loc[i,'cci']>0 and self.df.loc[i,'tsv']>0):
                self.df.loc[i,'signal'] = 1

            elif(self.df.loc[i,'macd_ema']<self.df.loc[i,'macd_signal'] and self.df.loc[i,'cci']<0 and self.df.loc[i,'tsv']<0):
                self.df.loc[i,'signal'] = -1

            else:
                self.df.loc[i,'signal'] = 0
        
        self.df.dropna(axis=0,inplace=True)
        print(f"the current df is {self.df} the macd status is {self.df['macd_ema'].iloc[-1]>self.df['macd_signal'].iloc[-1]} cci status is {self.df['cci'].iloc[-1]>0} tsv status is {self.df['tsv'].iloc[-1]>0} latest overall signal is {self.df['signal'].iloc[-1]}")

    def execute_signals(self):
        if self.df['signal'].iloc[-1] == 1:
            print("A buy signal will be placed")
            # self.broker.place_order()
        elif self.df['signal'].iloc[-1] == -1:
            print("A sell signal will be placed")
            # self.broker.place_order()
        else:
            print("No trade would be currently placed")

    def run(self):
        self.fetch_data()
        self.feature_engineering()
        self.generate_signals()
        self.execute_signals()

if __name__ == "__main__":
    s1 = Strategy1()
    s1.run()
