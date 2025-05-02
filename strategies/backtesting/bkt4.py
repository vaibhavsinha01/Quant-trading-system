# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import os

# class MFI_MACD_Backtester:
#     def __init__(self, data=None, initial_capital=100000):
#         """
#         Initialize the backtester with historical data and initial capital
        
#         Parameters:
#         -----------
#         data : pandas.DataFrame
#             Historical OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
#         initial_capital : float
#             Initial capital to start trading with
#         """
#         self.data = data
#         self.initial_capital = initial_capital
#         self.positions = pd.DataFrame()  # Will be initialized properly after data is loaded
#         self.portfolio = pd.DataFrame()  # Will be initialized properly after data is loaded
#         self.trades = []
#         self.commission_rate = 0.0005  # 0.05% commission rate (Angel typical rate)
#         self.slippage = 0.0002  # 0.02% slippage estimate

#     def load_data(self, filepath):
#         """
#         Load data from CSV file
        
#         Parameters:
#         -----------
#         filepath : str
#             Path to the CSV file containing OHLCV data
#         """
#         try:
#             self.data = pd.read_csv(filepath)
            
#             # Check if 'Date' or similar column exists to use as index
#             date_columns = [col for col in self.data.columns if col.lower() in ['date', 'time', 'timestamp', 'datetime']]
            
#             if date_columns:
#                 # Use the first found date column as index
#                 self.data['timestamp'] = pd.to_datetime(self.data[date_columns[0]])
#                 self.data.set_index('timestamp', inplace=True)
#             else:
#                 # If no date column found, convert the index to datetime if it's string
#                 if isinstance(self.data.index[0], str):
#                     self.data['timestamp'] = pd.to_datetime(self.data.index)
#                     self.data.set_index('timestamp', inplace=True)
#                 else:
#                     print("Warning: No timestamp column found, using default index")
            
#             # Ensure all required columns are present
#             required_columns = ['open', 'high', 'low', 'close', 'volume']
            
#             # Look for capitalized column names too
#             column_mapping = {}
#             for required in required_columns:
#                 if required not in self.data.columns:
#                     capitalized = required.capitalize()
#                     if capitalized in self.data.columns:
#                         column_mapping[capitalized] = required
            
#             # Rename columns if needed
#             if column_mapping:
#                 self.data = self.data.rename(columns=column_mapping)
            
#             # Check again for missing columns after renaming
#             missing_columns = [col for col in required_columns if col not in self.data.columns]
#             if missing_columns:
#                 print(f"Warning: Required columns {missing_columns} not found. Please ensure data contains: {required_columns}")
            
#             # Initialize positions and portfolio DataFrames with the proper index
#             self.positions = pd.DataFrame(index=self.data.index).fillna(0.0)
#             self.portfolio = pd.DataFrame(index=self.data.index).fillna(0.0)
            
#             print(f"Data loaded successfully with {len(self.data)} rows")
#             if hasattr(self.data.index, 'min') and hasattr(self.data.index, 'max'):
#                 print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
#             return True
        
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return False

#     def calculate_mfi(self, length=10):
#         """
#         Calculate Money Flow Index
        
#         Parameters:
#         -----------
#         length : int
#             The window length for MFI calculation
#         """
#         # Calculate typical price
#         typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        
#         # Calculate raw money flow
#         raw_money_flow = typical_price * self.data['volume']
        
#         # Get money flow direction
#         diff = typical_price.diff()
        
#         # Create positive and negative money flow series
#         positive_flow = pd.Series(np.where(diff > 0, raw_money_flow, 0), index=self.data.index)
#         negative_flow = pd.Series(np.where(diff < 0, raw_money_flow, 0), index=self.data.index)
        
#         # Sum positive and negative money flows over the period
#         positive_sum = positive_flow.rolling(window=length).sum()
#         negative_sum = negative_flow.rolling(window=length).sum()
        
#         # Calculate money flow ratio (handle division by zero)
#         money_ratio = np.where(negative_sum != 0, positive_sum / negative_sum, 100)
        
#         # Calculate MFI
#         self.data['mfi'] = 100 - (100 / (1 + money_ratio))

#     def calculate_macd(self, fast=12, slow=26, signal=9):
#         """
#         Calculate MACD indicator
        
#         Parameters:
#         -----------
#         fast : int
#             Fast EMA period
#         slow : int
#             Slow EMA period
#         signal : int
#             Signal line period
#         """
#         # Calculate MACD components
#         ema_fast = self.data['close'].ewm(span=fast, adjust=False).mean()
#         ema_slow = self.data['close'].ewm(span=slow, adjust=False).mean()
        
#         self.data['macd_ema'] = ema_fast - ema_slow
#         self.data['macd_signal'] = self.data['macd_ema'].ewm(span=signal, adjust=False).mean()
#         self.data['macd_hist'] = self.data['macd_ema'] - self.data['macd_signal']

#     def generate_signals(self):
#         """
#         Generate trading signals based on MFI divergence with MACD
#         """
#         # Ensure we have the required indicators
#         if 'mfi' not in self.data.columns:
#             self.calculate_mfi()
        
#         if 'macd_ema' not in self.data.columns:
#             self.calculate_macd()
        
#         # Initialize signal column
#         self.data['signal'] = 0
        
#         # Start from a safe index to avoid indexing errors
#         if len(self.data) < 3:
#             print("Not enough data to generate signals")
#             return
            
#         for i in range(2, len(self.data)):
#             # Get current row and indexes for previous rows
#             curr = self.data.iloc[i]
#             prev = self.data.iloc[i-1]
#             prev2 = self.data.iloc[i-2]
            
#             # Skip if any required values are NaN
#             if (pd.isna(curr['mfi']) or pd.isna(prev['mfi']) or pd.isna(prev2['mfi']) or
#                 pd.isna(curr['close']) or pd.isna(prev['close']) or pd.isna(prev2['close']) or
#                 pd.isna(curr['macd_ema']) or pd.isna(curr['macd_signal'])):
#                 continue
                
#             # Buy signal: MACD line crosses above signal line + MFI divergence
#             if (curr['macd_ema'] >= curr['macd_signal'] and 
#                 (curr['mfi'] >= prev['mfi'] and curr['close'] <= prev['close']) and 
#                 (prev['mfi'] >= prev2['mfi'] and prev['close'] <= prev2['close'])):
#                 self.data.iloc[i, self.data.columns.get_loc('signal')] = 1
            
#             # Sell signal: MACD line crosses below signal line + MFI divergence
#             elif (curr['macd_ema'] <= curr['macd_signal'] and 
#                  (curr['mfi'] <= prev['mfi'] and curr['close'] >= prev['close']) and 
#                  (prev['mfi'] <= prev2['mfi'] and prev['close'] >= prev2['close'])):
#                 self.data.iloc[i, self.data.columns.get_loc('signal')] = -1
        
#         print(f"Signals generated: {self.data['signal'].value_counts().to_dict()}")

#     def run_backtest(self, position_size=1.0):
#         """
#         Run backtest simulation
        
#         Parameters:
#         -----------
#         position_size : float
#             Size of position as a fraction of portfolio value (1.0 = 100%)
#         """
#         # Ensure signals are generated
#         if 'signal' not in self.data.columns:
#             self.generate_signals()
        
#         # Make sure we have valid data
#         if self.data is None or self.data.empty:
#             print("No data available for backtesting")
#             return
            
#         # Make sure we have a proper index
#         if not isinstance(self.data.index, pd.DatetimeIndex):
#             print("Warning: index is not a DatetimeIndex. Using numerical indexing.")
            
#         # Initialize portfolio and position tracking
#         self.positions = pd.DataFrame(index=self.data.index).fillna(0.0)
#         self.positions['position'] = 0
        
#         self.portfolio = pd.DataFrame(index=self.data.index).fillna(0.0)
#         self.portfolio['holdings'] = 0.0
#         self.portfolio['cash'] = self.initial_capital
#         self.portfolio['total'] = self.initial_capital
#         self.portfolio['returns'] = 0.0
        
#         self.trades = []
        
#         # Forward pass to determine positions based on signals
#         current_position = 0
#         entry_price = 0
#         entry_time = None
        
#         for i, row in self.data.iterrows():
#             # Skip if any required values are NaN
#             if pd.isna(row['signal']) or pd.isna(row['close']):
#                 continue
            
#             # Update portfolio value first, then check for new trades
#             if i > self.data.index[0]:  # Skip first row for returns calculation
#                 prev_value = self.portfolio.loc[self.data.index[self.data.index.get_loc(i)-1], 'total']
#                 self.portfolio.loc[i, 'returns'] = (self.portfolio.loc[i, 'total'] / prev_value) - 1
            
#             # Process signals
#             if row['signal'] == 1 and current_position <= 0:  # Buy signal
#                 # Close any existing short position
#                 if current_position < 0:
#                     exit_price = row['close'] * (1 + self.slippage)  # Account for slippage
#                     exit_commission = abs(current_position) * exit_price * self.commission_rate
#                     exit_value = abs(current_position) * exit_price - exit_commission
                    
#                     # Record the trade
#                     self.trades.append({
#                         'entry_time': entry_time,
#                         'exit_time': i,
#                         'entry_price': entry_price,
#                         'exit_price': exit_price,
#                         'position': current_position,
#                         'pnl': exit_value - (abs(current_position) * entry_price),
#                         'return': (exit_price / entry_price) - 1,
#                         'commission': exit_commission,
#                         'type': 'short'
#                     })
                
#                 # Enter new long position
#                 available_capital = self.portfolio.loc[i, 'cash']
#                 entry_price = row['close'] * (1 + self.slippage)  # Account for slippage
#                 position_value = available_capital * position_size
#                 shares_to_buy = position_value / entry_price
#                 commission = shares_to_buy * entry_price * self.commission_rate
                
#                 # Adjust for commission
#                 shares_to_buy = (position_value - commission) / entry_price
                
#                 # Update position and portfolio
#                 current_position = shares_to_buy
#                 entry_time = i
                
#                 self.portfolio.loc[i, 'cash'] -= (shares_to_buy * entry_price + commission)
#                 self.portfolio.loc[i, 'holdings'] = shares_to_buy * row['close']
            
#             elif row['signal'] == -1 and current_position >= 0:  # Sell signal
#                 # Close any existing long position
#                 if current_position > 0:
#                     exit_price = row['close'] * (1 - self.slippage)  # Account for slippage
#                     exit_commission = current_position * exit_price * self.commission_rate
#                     exit_value = current_position * exit_price - exit_commission
                    
#                     # Record the trade
#                     self.trades.append({
#                         'entry_time': entry_time,
#                         'exit_time': i,
#                         'entry_price': entry_price,
#                         'exit_price': exit_price,
#                         'position': current_position,
#                         'pnl': exit_value - (current_position * entry_price),
#                         'return': (exit_price / entry_price) - 1,
#                         'commission': exit_commission,
#                         'type': 'long'
#                     })
                    
#                     # Update cash from closing position
#                     self.portfolio.loc[i, 'cash'] += exit_value
                
#                 # Enter new short position if shorting is allowed
#                 available_capital = self.portfolio.loc[i, 'cash']
#                 entry_price = row['close'] * (1 - self.slippage)  # Account for slippage
#                 position_value = available_capital * position_size
#                 shares_to_short = position_value / entry_price
#                 commission = shares_to_short * entry_price * self.commission_rate
                
#                 # Adjust for commission
#                 shares_to_short = (position_value - commission) / entry_price
                
#                 # Update position and portfolio
#                 current_position = -shares_to_short
#                 entry_time = i
                
#                 # No immediate cash change for short selling
#                 self.portfolio.loc[i, 'holdings'] = -shares_to_short * row['close']
            
#             # Update positions for current timestamp
#             self.positions.loc[i, 'position'] = current_position
            
#             # Update portfolio value
#             self.portfolio.loc[i, 'holdings'] = current_position * row['close']
#             self.portfolio.loc[i, 'total'] = self.portfolio.loc[i, 'cash'] + self.portfolio.loc[i, 'holdings']
        
#         # Forward fill portfolio and position values
#         self.positions = self.positions.fillna(method='ffill')
#         self.portfolio = self.portfolio.fillna(method='ffill')
        
#         # Calculate cumulative returns
#         self.portfolio['cumulative_return'] = (1 + self.portfolio['returns']).cumprod() - 1
        
#         print(f"Backtest completed. Final portfolio value: {self.portfolio['total'].iloc[-1]:.2f}")
#         print(f"Total return: {(self.portfolio['total'].iloc[-1] / self.initial_capital - 1) * 100:.2f}%")
#         print(f"Number of trades: {len(self.trades)}")

#     def calculate_performance_metrics(self):
#         """
#         Calculate performance metrics for the strategy
        
#         Returns:
#         --------
#         dict: Dictionary containing performance metrics
#         """
#         if self.portfolio.empty:
#             print("No backtest results available. Run backtest first.")
#             return {}
        
#         # Basic performance metrics
#         total_return = (self.portfolio['total'].iloc[-1] / self.initial_capital) - 1
#         daily_returns = self.portfolio['returns'].dropna()
        
#         # Annualized metrics (assuming daily data)
#         trading_days_per_year = 252
#         ann_return = (1 + total_return) ** (trading_days_per_year / len(daily_returns)) - 1
#         ann_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
#         sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0
        
#         # Drawdown analysis
#         cum_returns = (1 + daily_returns).cumprod()
#         running_max = cum_returns.cummax()
#         drawdown = (cum_returns / running_max) - 1
#         max_drawdown = drawdown.min()
        
#         # Trade statistics
#         if self.trades:
#             trades_df = pd.DataFrame(self.trades)
#             winning_trades = trades_df[trades_df['pnl'] > 0]
#             losing_trades = trades_df[trades_df['pnl'] <= 0]
            
#             win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
#             avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
#             avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
#             profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
            
#             # Average holding period
#             trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 86400  # Convert to days
#             avg_holding_period = trades_df['holding_period'].mean()
#         else:
#             win_rate = 0
#             avg_win = 0
#             avg_loss = 0
#             profit_factor = 0
#             avg_holding_period = 0
        
#         metrics = {
#             'total_return': total_return,
#             'annualized_return': ann_return,
#             'annualized_volatility': ann_volatility,
#             'sharpe_ratio': sharpe_ratio,
#             'max_drawdown': max_drawdown,
#             'win_rate': win_rate,
#             'avg_win': avg_win,
#             'avg_loss': avg_loss,
#             'profit_factor': profit_factor,
#             'number_of_trades': len(self.trades),
#             'avg_holding_period': avg_holding_period,
#             'final_portfolio_value': self.portfolio['total'].iloc[-1]
#         }
        
#         return metrics

#     def plot_results(self, save_path=None):
#         """
#         Plot backtest results
        
#         Parameters:
#         -----------
#         save_path : str, optional
#             Path to save the plot
#         """
#         if self.portfolio.empty:
#             print("No backtest results available. Run backtest first.")
#             return
        
#         # Create figure with subplots
#         fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1.5, 1]})
        
#         # Set style
#         sns.set_style('whitegrid')
        
#         # 1. Portfolio value over time
#         self.portfolio['total'].plot(ax=axes[0], color='blue', linewidth=2)
#         axes[0].set_title('Portfolio Value Over Time', fontsize=14)
#         axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
#         axes[0].set_xlabel('')
#         axes[0].grid(True)
        
#         # 2. Price chart with buy/sell signals
#         self.data['close'].plot(ax=axes[1], color='black', linewidth=1.5)
        
#         # Plot buy signals
#         buy_signals = self.data[self.data['signal'] == 1]
#         axes[1].scatter(buy_signals.index, buy_signals['close'], 
#                       marker='^', color='green', s=100, label='Buy Signal')
        
#         # Plot sell signals
#         sell_signals = self.data[self.data['signal'] == -1]
#         axes[1].scatter(sell_signals.index, sell_signals['close'], 
#                        marker='v', color='red', s=100, label='Sell Signal')
        
#         axes[1].set_title('Price Chart with Signals', fontsize=14)
#         axes[1].set_ylabel('Price ($)', fontsize=12)
#         axes[1].set_xlabel('')
#         axes[1].grid(True)
#         axes[1].legend(loc='best')
        
#         # 3. Drawdown chart
#         daily_returns = self.portfolio['returns'].dropna()
#         cum_returns = (1 + daily_returns).cumprod()
#         running_max = cum_returns.cummax()
#         drawdown = (cum_returns / running_max) - 1
        
#         drawdown.plot(ax=axes[2], color='red', linewidth=1.5)
#         axes[2].fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
#         axes[2].set_title('Drawdown', fontsize=14)
#         axes[2].set_ylabel('Drawdown (%)', fontsize=12)
#         axes[2].set_xlabel('Date', fontsize=12)
#         axes[2].grid(True)
        
#         # Adjust layout
#         plt.tight_layout()
        
#         # Save plot if path provided
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Plot saved to {save_path}")
        
#         plt.show()

#     def save_results(self, output_dir='backtest_results'):
#         """
#         Save backtest results to files
        
#         Parameters:
#         -----------
#         output_dir : str
#             Directory to save results
#         """
#         if self.portfolio.empty:
#             print("No backtest results available. Run backtest first.")
#             return
        
#         # Create output directory if it doesn't exist
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         # Generate timestamp for filenames
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Save portfolio data
#         portfolio_file = f"{output_dir}/portfolio_{timestamp}.csv"
#         self.portfolio.to_csv(portfolio_file)
        
#         # Save trade data
#         if self.trades:
#             trades_file = f"{output_dir}/trades_{timestamp}.csv"
#             trades_df = pd.DataFrame(self.trades)
#             trades_df.to_csv(trades_file, index=False)
        
#         # Save performance metrics
#         metrics = self.calculate_performance_metrics()
#         metrics_file = f"{output_dir}/metrics_{timestamp}.csv"
#         pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        
#         # Save plot
#         plot_file = f"{output_dir}/plot_{timestamp}.png"
#         self.plot_results(save_path=plot_file)
        
#         print(f"Results saved to {output_dir} directory")


# # Example usage
# if __name__ == "__main__":
#     try:
#         print("Starting backtesting process...")
        
#         # Initialize backtester with initial capital
#         backtester = MFI_MACD_Backtester(initial_capital=100000)
#         print("Backtester initialized successfully.")
        
#         # Load data from CSV file
#         data_file = 'features.csv'
#         print(f"Attempting to load data from {data_file}...")
        
#         if backtester.load_data(data_file):  # Use the CSV saved by your strategy
#             print("Data loaded successfully.")
            
#             # Make sure required columns exist
#             required_columns = ['open', 'high', 'low', 'close', 'volume']
#             missing_columns = [col for col in required_columns if col not in backtester.data.columns]
            
#             if missing_columns:
#                 print(f"Warning: Missing required columns: {missing_columns}")
#                 print("Available columns:", backtester.data.columns.tolist())
                
#                 # Try to find appropriate columns
#                 if 'Open' in backtester.data.columns and 'open' not in backtester.data.columns:
#                     print("Renaming capitalized column names to lowercase...")
#                     backtester.data = backtester.data.rename(columns={
#                         'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
#                     })
            
#             # Calculate indicators
#             print("Calculating MFI indicator...")
#             backtester.calculate_mfi(length=10)  # Use the same parameters as your strategy
            
#             print("Calculating MACD indicator...")
#             backtester.calculate_macd()
            
#             # Generate signals and run backtest
#             print("Generating trading signals...")
#             backtester.generate_signals()
            
#             print("Running backtest simulation...")
#             backtester.run_backtest(position_size=0.8)  # Use 80% of available capital for each position
            
#             # Display and save results
#             print("Calculating performance metrics...")
#             metrics = backtester.calculate_performance_metrics()
            
#             print("\n--- Performance Metrics ---")
#             for key, value in metrics.items():
#                 print(f"{key}: {value}")
            
#             print("Generating performance plots...")
#             backtester.plot_results()
            
#             print("Saving backtest results...")
#             backtester.save_results()
            
#             print("Backtesting process completed successfully.")
#         else:
#             print(f"Failed to load data from {data_file}. Please check if the file exists and contains valid data.")
    
#     except Exception as e:
#         import traceback
#         print(f"Error during backtesting: {e}")
#         print("Detailed error information:")
#         traceback.print_exc()

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# === Custom Indicator Functions ===

def calculate_mfi(df, length=14):
    """
    Money Flow Index (MFI)
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    diff = typical_price.diff()

    positive_flow = pd.Series(np.where(diff > 0, raw_money_flow, 0), index=df.index)
    negative_flow = pd.Series(np.where(diff < 0, raw_money_flow, 0), index=df.index)

    positive_sum = positive_flow.rolling(window=length).sum()
    negative_sum = negative_flow.rolling(window=length).sum()

    money_ratio = np.where(negative_sum != 0, positive_sum / negative_sum, 100)
    mfi = 100 - (100 / (1 + money_ratio))

    return pd.Series(mfi, index=df.index)


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence (MACD)
    Returns MACD line, Signal line, and Histogram
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# === Strategy ===
from backtesting.lib import SignalStrategy, TrailingStrategy
import numpy as np

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    return atr

class MFI_MACD_Strategy(Strategy):
    atr_length = 14  # You can tune this if needed

    def init(self):
        self.df = self.data.df.copy()
        self.df['mfi'] = calculate_mfi(self.df, length=10)
        self.df['macd'], self.df['macd_signal'], self.df['macd_hist'] = calculate_macd(self.df)
        self.df['atr'] = calculate_atr(self.df, length=self.atr_length)

        self.entry_price = None
        self.atr_at_entry = None

    def next(self):
        i = len(self.df) - len(self.data.Close)
        if len(self.data.Close) < 3 or i < 2:
            return

        mfi = self.df['mfi'].iloc[i]
        prev_mfi = self.df['mfi'].iloc[i - 1]
        prev2_mfi = self.df['mfi'].iloc[i - 2]

        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        prev2_close = self.data.Close[-3]

        macd = self.df['macd'].iloc[i]
        signal = self.df['macd_signal'].iloc[i]
        atr = self.df['atr'].iloc[i]

        # Exit: TP/SL logic
        if self.position:
            if self.position.is_long:
                tp = self.entry_price + 3 * self.atr_at_entry
                sl = self.entry_price - 2 * self.atr_at_entry
                if close >= tp or close <= sl or macd < signal:
                    self.position.close()
            elif self.position.is_short:
                tp = self.entry_price - 3 * self.atr_at_entry
                sl = self.entry_price + 2 * self.atr_at_entry
                if close <= tp or close >= sl or macd > signal:
                    self.position.close()
            return  # Skip re-entry if in position

        # Entry: Long
        # Entry: Long
        if (
            (mfi <= 40) and
            (mfi >= prev_mfi and close <= prev_close) and
            (prev_mfi >= prev2_mfi and prev_close <= prev2_close)
        ):
            sl = close + 2 * atr
            tp = close - 3 * atr
            self.sell(sl=sl, tp=tp)
            self.entry_price = close
            self.atr_at_entry = atr

        # Entry: Short
        elif (
            (mfi >= 60) and
            (mfi <= prev_mfi and close >= prev_close) and
            (prev_mfi <= prev2_mfi and prev_close >= prev2_close)
        ):
            sl = close - 2 * atr
            tp = close + 3 * atr
            self.buy(sl=sl, tp=tp)
            self.entry_price = close
            self.atr_at_entry = atr


# === Run Backtest ===

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\qts\features.csv')
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    bt = Backtest(df, MFI_MACD_Strategy, cash=100000)
    results = bt.run()
    print(results)
    bt.plot()

