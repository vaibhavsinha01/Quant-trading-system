from angel import AngelBrokerWrapper
from datetime import datetime, timedelta
import pandas as pd
import time
import traceback

class TradeWilliamR:
    def __init__(self):
        try:
            # self.api_key = 'AYq866MN'
            self.api_key = '7NucMs8X'
            # self.username = 'N60066209'
            self.username = 'AAAI566003'
            # self.password = '1010'
            self.password = '1111'
            # self.token = "TJYZILIYM56MEDGUZXXKL7QOKE"
            self.token = 'W2SNZLD5UY4SVTP34PWKYGQQZ4'
            self.symbol_token = '3045'
            self.tradingsymbol = 'SBIN-EQ'
            self.exchange = 'NSE'
            self.interval = 'ONE_DAY'
            self.lookback_days = 100
            self.in_position = False
            self.entry_price = None
            print("[INIT] TradeWilliamR initialized.")
        except Exception as e:
            print("[ERROR] Initialization failed:", e)
            traceback.print_exc()

    def indicators_calc(self, df):
        try:
            print("[STEP] Calculating indicators...")
            period = 14
            df['highest_high'] = df['high'].rolling(window=period).max()
            df['lowest_low'] = df['low'].rolling(window=period).min()
            df['willr'] = -100 * ((df['highest_high'] - df['close']) / (df['highest_high'] - df['lowest_low']))
            print("[DONE] Williams %R calculated.")
            return df
        except Exception as e:
            print("[ERROR] Failed to calculate indicators:", e)
            traceback.print_exc()
            return df

    def signal_generation(self, df):
        try:
            print("[STEP] Generating signals...")
            df['entry_signal'] = False
            df['exit_signal'] = False
            df['willr_prev'] = df['willr'].shift(1)
            df['close_prev'] = df['close'].shift(1)
            df['high_prev'] = df['high'].shift(1)

            df['entry_signal'] = (df['willr_prev'] > -90) & (df['willr'] <= -90)
            df['exit_signal'] = ((df['willr_prev'] < -30) & (df['willr'] >= -30)) | (df['close'] > df['high_prev'])

            print("[DONE] Signals generated.")
            return df
        except Exception as e:
            print("[ERROR] Failed to generate signals:", e)
            traceback.print_exc()
            return df

    def place_buy_order(self, price):
        try:
            print(f"[ORDER] Preparing BUY order for {self.tradingsymbol}.")
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": str(self.tradingsymbol),
                "symboltoken": str(self.symbol_token),
                "transactiontype": "BUY",
                "exchange": str(self.exchange),
                "ordertype": "MARKET",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": "0",
                "quantity": str(1)
            }
            response = self.broker.place_order(orderparams)
            print(f"[ORDER] BUY placed — Response: {response}")
        except Exception as e:
            print("[ERROR] Failed to place BUY order:", e)
            traceback.print_exc()

    def place_sell_order(self, price):
        try:
            print(f"[ORDER] Preparing SELL order for {self.tradingsymbol}.")
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": str(self.tradingsymbol),
                "symboltoken": str(self.symbol_token),
                "transactiontype": "SELL",
                "exchange": str(self.exchange),
                "ordertype": "MARKET",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": "0",
                "quantity": str(1)
            }
            response = self.broker.place_order(orderparams)
            print(f"[ORDER] SELL placed — Response: {response}")
        except Exception as e:
            print("[ERROR] Failed to place SELL order:", e)
            traceback.print_exc()

    def execute_trades(self, df):
        try:
            print("[STEP] Evaluating trade signals...")
            latest = df.iloc[-1]
            print(f"[DATA] Latest candle: {latest[['timestamp', 'close', 'willr', 'entry_signal', 'exit_signal']]}")

            current_close = latest['close']
            high = latest['high']
            low = latest['low']

            if not self.in_position and latest['entry_signal']:
                print(f"[ENTRY] Signal detected — BUY at {current_close:.2f}")
                self.entry_price = current_close
                self.place_buy_order(price=current_close)
                self.in_position = True
                print(f"[STATUS] Position opened at {self.entry_price:.2f} — TP: {high:.2f}, SL: {low:.2f}")

            elif self.in_position:
                if latest['exit_signal']:
                    print(f"[EXIT] Signal detected — SELL at {current_close:.2f}")
                    self.place_sell_order(price=current_close)
                    self.in_position = False
                    print("[STATUS] Position closed.")
                else:
                    print(f"[INFO] Holding — Entry: {self.entry_price:.2f}, Current: {current_close:.2f}, TP: {high:.2f}, SL: {low:.2f}")
            else:
                print("[INFO] No entry signal and no open position.")
        except Exception as e:
            print("[ERROR] Trade execution failed:", e)
            traceback.print_exc()

    def run(self):
        try:
            print("[INFO] Starting run cycle...")
            print("[STEP] Connecting to broker...")
            self.broker = AngelBrokerWrapper(
                api_key=self.api_key,
                username=self.username,
                password=self.password,
                token=self.token
            )
            self.broker.connect()
            print("[DONE] Broker connected.")

            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9,minute=30,second=0,microsecond=0)
            start_time = market_open_time - timedelta(days=100)

            print(f"[STEP] Fetching data from {start_time} to {current_time}...")

            df = self.broker.get_candle_data(
                exchange=self.exchange,
                symboltoken=self.symbol_token,
                interval=self.interval,
                fromdate=start_time.strftime("%Y-%m-%d %H:%M"),
                todate=current_time.strftime("%Y-%m-%d %H:%M")
            )

            print(f"[DATA] Fetched {len(df)} rows of candle data.")
            df = pd.DataFrame(df)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = self.indicators_calc(df)
            df = self.signal_generation(df)
            self.execute_trades(df)

            print("[DONE] Run cycle complete.")

        except Exception as e:
            print("[ERROR] Run cycle failed:", e)
            traceback.print_exc()


if __name__ == "__main__":
    session = TradeWilliamR()
    while True:
        session.run()
        print("[SLEEP] Sleeping for 60 seconds...\n")
        time.sleep(60)
