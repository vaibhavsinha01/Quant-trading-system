from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EndOfMonthStrategy(Strategy):
    buy_day = 28  
    sell_day = 3  
    
    def init(self):
        try:
            self.current_day = self.I(lambda: pd.Series(self.data.index.day, index=self.data.index), name='day')
            logger.info(f"Strategy initialized. Data spans from {self.data.index[0]} to {self.data.index[-1]}")
            logger.info(f"There are {len(self.data.index)} data points")
        except Exception as e:
            logger.error(f"Error in init: {str(e)}")
    
    def next(self):
        try:
            current_day = self.data.index[-1].day
            
            if current_day == self.buy_day:
                logger.info(f"Buy signal at {self.data.index[-1]} - Day {current_day}")
                
            if current_day == self.sell_day:
                logger.info(f"Sell signal at {self.data.index[-1]} - Day {current_day}")
            
            if current_day == self.buy_day and not self.position:
                self.buy()
                logger.info(f"BUY executed at {self.data.Close[-1]}")
            
            # Sell condition: day is the sell_day (3rd) and we have a position
            elif current_day == self.sell_day and self.position:
                self.position.close()
                logger.info(f"SELL executed at {self.data.Close[-1]}")
        except Exception as e:
            logger.error(f"Error in next: {str(e)}")


# Run the backtest
if __name__ == '__main__':
    try:
        # Load data
        logger.info("Loading data...")
        # data = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\qts\strategies\reliance_fetched_data_1year_15m.csv")
        data = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\ribbon_1\t3_ribbon_3rd_march (3)\t3_ribbon_3rd_march\t3-ribbon\normal_indicator.csv")
        # Parse datetime column
        logger.info("Processing datetime...")
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Rename columns to match backtesting.py requirements
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Set datetime as index
        data.set_index('datetime', inplace=True)
        
        # Print out data info
        logger.info(f"Data spans from {data.index.min()} to {data.index.max()}")
        logger.info(f"Days in data: {sorted(data.index.day.unique())}")
        
        # Additional validation check
        days_in_data = set(data.index.day.unique())
        strategy_buy_day = EndOfMonthStrategy.buy_day
        strategy_sell_day = EndOfMonthStrategy.sell_day
        
        logger.info(f"Buy day {strategy_buy_day} in data: {strategy_buy_day in days_in_data}")
        logger.info(f"Sell day {strategy_sell_day} in data: {strategy_sell_day in days_in_data}")
        
        # Optionally ensure we have at least one of each signal day
        if strategy_buy_day not in days_in_data:
            logger.warning(f"No data for buy day {strategy_buy_day} in dataset!")
        if strategy_sell_day not in days_in_data:
            logger.warning(f"No data for sell day {strategy_sell_day} in dataset!")
        
        # Run the backtest
        logger.info("Running backtest...")
        bt = Backtest(data, EndOfMonthStrategy, cash=10000)
        stats = bt.run()
        
        # Print the results
        print(stats)
        
        # Plot the results
        bt.plot()
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()