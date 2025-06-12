from angel import AngelBrokerWrapper
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
import creds
from datetime import datetime,timedelta
import time

class SupertrendEma:
    def __init__(self):
        self.df = None

    def fetch_data(self):
        self.df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\qts\features.csv")

    def indicator_calculation(self):
        self.df['ema1_10'] = EMAIndicator(close=self.df['close'],window=10)
        self.df['ema2_10'] = EMAIndicator(close=self.df['ema1'],window=10)
        self.df['ema1_20'] = EMAIndicator(close=self.df['close'],window=20)
        self.df['ema2_20'] = EMAIndicator(close=self.df['ema1'],window=20)
        self.df['atr'] = AverageTrueRange(high=self.df['high'],low=self.df['low'],close=self.df['close'],window=10).average_true_range()
        self.df['upper'] = self.df['close']+3*self.df['atr']
        self.df['lower'] = self.df['close']-3*self.df['atr']
        self.df['super_signal'] = 0
        self.df['trend_signal'] = 0

        for i in range(len(self.df)):
            if(self.df.loc[i,'close']<self.df.loc[i-1,'lower']):
                self.df.loc[i,'super_signal'] = 1
            elif(self.df.loc[i,'close']>self.df.loc[i-1,'upper']):
                self.df.loc[i,'super_signal'] = -1
            else:
                self.df.loc[i,'super_signal'] = 0

        for i in range(len(self.df)):
            if(self.df.loc[i,'ema2_10']>self.df.loc[i-1,'ema2_10'] or self.df.loc[i,'ema2_20']>self.df.loc[i-1,'ema2_20']):
                self.df.loc[i,'trend_signal'] = 1
            elif(self.df.loc[i,'ema2_10']>self.df.loc[i-1,'ema2_10'] or self.df.loc[i,'ema2_20']>self.df.loc[i-1,'ema2_20']):
                self.df.loc[i,'trend_signal'] = -1
            else:
                self.df.loc[i,'trend_signal'] = 0
    
    def place_order_with_tp_sl(self, transaction_type, quantity, price, stoploss, target):
        order = self.broker.place_order(
            variety='NORMAL',
            tradingsymbol='RELIANCE-EQ',
            symboltoken='3045',
            exchange='NSE',
            transactiontype=transaction_type,
            ordertype='MARKET',
            producttype='INTRADAY',
            duration='DAY',
            price=price,
            squareoff=0,
            stoploss=0,
            quantity=quantity
        )

        order_id = order['data']['orderid']
        print(f"Main order placed: {order_id}")

        # Place stoploss order
        sl_price = price - stoploss if transaction_type == 'BUY' else price + stoploss
        sl_order = self.broker.place_order(
            variety='STOPLOSS',
            tradingsymbol='RELIANCE-EQ',
            symboltoken='3045',
            exchange='NSE',
            transactiontype='SELL' if transaction_type == 'BUY' else 'BUY',
            ordertype='SL',
            producttype='INTRADAY',
            duration='DAY',
            price=sl_price,
            triggerprice=sl_price,
            quantity=quantity
        )
        print(f"SL order placed: {sl_order['data']['orderid']}")

        # Place target order
        tp_price = price + target if transaction_type == 'BUY' else price - target
        tp_order = self.broker.place_order(
            variety='NORMAL',
            tradingsymbol='RELIANCE-EQ',
            symboltoken='3045',
            exchange='NSE',
            transactiontype='SELL' if transaction_type == 'BUY' else 'BUY',
            ordertype='LIMIT',
            producttype='INTRADAY',
            duration='DAY',
            price=tp_price,
            quantity=quantity
        )
        print(f"TP order placed: {tp_order['data']['orderid']}")
    
    def close_related_tp_sl(self):
        print('if either the tp/sl is hitted then the other would automatically close')
    
    def trade_execution(self):
        latest = self.df.iloc[-1]

        if(latest['trend_signal'] == 1 and latest['super_signal'] == 1):
            print('buy order is being placed')
            res = self.place_order_with_tp_sl(transaction_type='BUY',quantity=4,price=latest['close'],stoploss=latest['close']-1.5*latest['atr'],target=latest['close']+2.5*latest['atr'])
            print(res)
        elif(latest['trend_signal'] == -1 and latest['super_signal'] == -1):
            print('sell order is being placed')
            res = self.place_order_with_tp_sl(transaction_type='SELL',quantity=4,price=latest['close'],stoploss=latest['close']+1.5*latest['atr'],target=latest['close']+2.5*latest['atr'])
            print(res)
        else:
            print("No order would be placed now.")
    
    def run(self):
        self.broker = AngelBrokerWrapper(api_key=creds.api_key,username=creds.username,password=creds.password,token=creds.token,correlation_id='abcde')
        self.broker.connect()
        orderbook = self.broker.get_orderbook()
        orderbook_data = orderbook['data']
        self.current_df = pd.DataFrame(orderbook_data)
        self.current_df.to_csv('orders_open.csv')
        while True:
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9,minute=30,second=0,microsecond=0)
            start_time = market_open_time - timedelta(days=100)
            self.df = self.broker.get_candle_data(exchange='NSE',symboltoken='3045',interval='ONE_DAY',fromdate=start_time,todate=current_time)
            self.indicator_calculation()
            self.close_related_tp_sl() # implement this
            self.trade_execution()
            time.sleep(60)

if __name__ == "__main__":
    supertrendema = SupertrendEma()
    supertrendema.run()

