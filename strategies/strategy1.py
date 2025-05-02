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
        self.df['atr_roc'],self.df['atr_roc_ema'],self.df['directionality_vol'],self.df['vol_strength'] = self.volatI.directional_volatility(self.df,window=10)
        self.df['cci'] = self.momenI.cci(self.df)
        self.df['tsv'] = self.volumI.tsv(self.df)
        self.df['macd_ema'],self.df['macd_signal'],self.df['histogram'] = self.trendI.macd(self.df)
        self.df['uo'] = self.momenI.uo(self.df)
        print(self.df)
        self.df.to_csv("features.csv")

    def generate_signals(self):
        for i in range(len(self.df)):
            if(self.df.loc[i,'macd_ema']>self.df.loc[i,'macd_signal'] and self.df.loc[i,'uo']>70 and self.df.loc[i,'tsv']>0):
                self.df.loc[i,'signal'] = 1

            elif(self.df.loc[i,'macd_ema']<self.df.loc[i,'macd_signal'] and self.df.loc[i,'uo']<30 and self.df.loc[i,'tsv']<0):
                self.df.loc[i,'signal'] = -1

            else:
                self.df.loc[i,'signal'] = 0
        
        self.df.dropna(axis=0,inplace=True)
        print(f"the current df is {self.df} the macd status is {self.df['macd_ema'].iloc[-1]>self.df['macd_signal'].iloc[-1]} uo status is {self.df['uo'].iloc[-1]<30 or self.df['uo'].iloc[-1]>70} tsv status is {self.df['tsv'].iloc[-1]>0} latest overall signal is {self.df['signal'].iloc[-1]}")

    def execute_signals(self): # the entire logic tm/rm/pm would be used here this is a simple application
        if self.df['signal'].iloc[-1] == 1:
            print("A buy signal will be placed")
            # self.broker.place_order()
        elif self.df['signal'].iloc[-1] == -1:
            print("A sell signal will be placed")
            # self.broker.place_order()
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
