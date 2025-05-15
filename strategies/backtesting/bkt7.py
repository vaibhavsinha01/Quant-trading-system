import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_supertrend(df, atr_period=14, multiplier=2.5):
    """
    Calculate Supertrend indicator using HL/2 approach
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'High', 'Low', 'Close' columns
    atr_period : int
        Period for ATR calculation
    multiplier : float
        Multiplier for ATR to set bands
        
    Returns:
    --------
    pandas.DataFrame with Supertrend values and signals
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Calculate median price (HL/2)
    df['median_price'] = (pd.Series(df['High']) + pd.Series(df['Low'])) / 2
    
    # Calculate ATR
    df['tr1'] = abs(pd.Series(df['High']) - pd.Series(df['Low']))
    df['tr2'] = abs(pd.Series(df['High']) - pd.Series(df['Close']).shift(1))
    df['tr3'] = abs(pd.Series(df['Low']) - pd.Series(df['Close']).shift(1))
    # df['tr'] = pd.Series(df[['tr1', 'tr2', 'tr3']]).max(axis=1)
    # df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['tr'] = pd.concat([df['tr1'], df['tr2'], df['tr3']], axis=1).max(axis=1)

    df['atr'] = pd.Series(df['tr']).rolling(window=atr_period).mean()
    
    # Calculate bands
    df['basic_upper_band'] = df['median_price'] + (multiplier * df['atr'])
    df['basic_lower_band'] = df['median_price'] - (multiplier * df['atr'])
    
    # Initialize Supertrend columns
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 0  # 1 for uptrend, -1 for downtrend
    df['final_upper_band'] = 0.0
    df['final_lower_band'] = 0.0
    
    # Calculate Supertrend
    for i in range(atr_period, len(df)):
        if i == atr_period:
            df.loc[df.index[i], 'final_upper_band'] = df.loc[df.index[i], 'basic_upper_band']
            df.loc[df.index[i], 'final_lower_band'] = df.loc[df.index[i], 'basic_lower_band']
            
            if df.loc[df.index[i], 'Close'] <= df.loc[df.index[i], 'final_upper_band']:
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_upper_band']
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower_band']
                df.loc[df.index[i], 'supertrend_direction'] = 1
        else:
            # Calculate final upper band
            if (df.loc[df.index[i-1], 'final_upper_band'] < df.loc[df.index[i], 'basic_upper_band'] or 
                df.loc[df.index[i-1], 'Close'] > df.loc[df.index[i-1], 'final_upper_band']):
                df.loc[df.index[i], 'final_upper_band'] = df.loc[df.index[i], 'basic_upper_band']
            else:
                df.loc[df.index[i], 'final_upper_band'] = df.loc[df.index[i-1], 'final_upper_band']
            
            # Calculate final lower band
            if (df.loc[df.index[i-1], 'final_lower_band'] > df.loc[df.index[i], 'basic_lower_band'] or 
                df.loc[df.index[i-1], 'Close'] < df.loc[df.index[i-1], 'final_lower_band']):
                df.loc[df.index[i], 'final_lower_band'] = df.loc[df.index[i], 'basic_lower_band']
            else:
                df.loc[df.index[i], 'final_lower_band'] = df.loc[df.index[i-1], 'final_lower_band']
            
            # Determine Supertrend value and direction
            if (df.loc[df.index[i-1], 'supertrend_direction'] == -1 and 
                df.loc[df.index[i], 'Close'] > df.loc[df.index[i], 'final_upper_band']):
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower_band']
                df.loc[df.index[i], 'supertrend_direction'] = 1
            elif (df.loc[df.index[i-1], 'supertrend_direction'] == 1 and 
                  df.loc[df.index[i], 'Close'] < df.loc[df.index[i], 'final_lower_band']):
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_upper_band']
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i-1], 'supertrend']
                df.loc[df.index[i], 'supertrend_direction'] = df.loc[df.index[i-1], 'supertrend_direction']
    
    # Generate buy/sell signals
    df['signal'] = 0
    df.loc[(df['supertrend_direction'] == 1) & (pd.Series(df['supertrend_direction']).shift(1) == -1), 'signal'] = 1  # Buy signal
    df.loc[(df['supertrend_direction'] == -1) & (pd.Series(df['supertrend_direction']).shift(1) == 1), 'signal'] = -1  # Sell signal
    
    return df

def calculate_double_ema(df, period1=10, period2=20):
    """
    Calculate EMA of EMA for confirmation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Close' price
    period1, period2 : int
        EMA periods
        
    Returns:
    --------
    Updated DataFrame with EMA of EMA values
    """
    df = df.copy()
    
    # Calculate EMA of EMA for period1
    df[f'ema_{period1}'] = pd.Series(df['Close']).ewm(span=period1, adjust=False).mean()
    df[f'ema_ema_{period1}'] = pd.Series(df[f'ema_{period1}']).ewm(span=period1, adjust=False).mean()
    
    # Calculate EMA of EMA for period2
    df[f'ema_{period2}'] = pd.Series(df['Close']).ewm(span=period2, adjust=False).mean()
    df[f'ema_ema_{period2}'] = pd.Series(df[f'ema_{period2}']).ewm(span=period2, adjust=False).mean()
    
    # Calculate slopes of EMAs
    df[f'slope_ema_ema_{period1}'] = pd.Series(df[f'ema_ema_{period1}']) - pd.Series(df[f'ema_ema_{period1}']).shift(1)
    df[f'slope_ema_ema_{period2}'] = pd.Series(df[f'ema_ema_{period2}']) - pd.Series(df[f'ema_ema_{period2}']).shift(1)
    
    return df

def backtest_supertrend_strategy(df, initial_capital=100000, initial_position_size=16):
    """
    Backtest Supertrend strategy with position sizing and trailing stop-loss
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Supertrend values and signals
    initial_capital : float
        Starting capital for the backtest
    initial_position_size : int
        Initial position size
        
    Returns:
    --------
    pandas.DataFrame with trade results and performance metrics
    """
    df = df.copy()
    
    # Initialize columns for backtesting
    df['position'] = 0  # Current position size
    df['tp'] = np.nan    # Take profit level
    df['sl'] = np.nan    # Stop loss level
    df['trailing_sl'] = np.nan  # Trailing stop loss
    df['trade_active'] = False  # Flag to track active trades
    df['half_tp'] = np.nan  # 50% of take profit for scaling
    df['quarter_tp'] = np.nan  # 25% of take profit for scaling
    df['scaled_in'] = 0  # Track scaling levels (0: no scaling, 1: first scale, 2: second scale)
    
    # Performance tracking
    df['capital'] = initial_capital
    df['pnl'] = 0.0
    
    # Track entry price for calculating profit/loss
    entry_price = 0
    current_position = 0
    
    for i in range(2, len(df)):
        prev_idx = df.index[i-1]
        curr_idx = df.index[i]
        
        # Check EMA confirmation (slope > 0 for either of the EMAs)
        ema10_slope_positive = df.loc[prev_idx, 'slope_ema_ema_10'] > 0
        ema20_slope_positive = df.loc[prev_idx, 'slope_ema_ema_20'] > 0
        ema_confirmation = ema10_slope_positive or ema20_slope_positive
        
        # Copy previous values
        df.loc[curr_idx, 'position'] = df.loc[prev_idx, 'position']
        df.loc[curr_idx, 'capital'] = df.loc[prev_idx, 'capital']
        df.loc[curr_idx, 'trade_active'] = df.loc[prev_idx, 'trade_active']
        df.loc[curr_idx, 'tp'] = df.loc[prev_idx, 'tp']
        df.loc[curr_idx, 'sl'] = df.loc[prev_idx, 'sl']
        df.loc[curr_idx, 'trailing_sl'] = df.loc[prev_idx, 'trailing_sl']
        df.loc[curr_idx, 'half_tp'] = df.loc[prev_idx, 'half_tp']
        df.loc[curr_idx, 'quarter_tp'] = df.loc[prev_idx, 'quarter_tp']
        df.loc[curr_idx, 'scaled_in'] = df.loc[prev_idx, 'scaled_in']
        
        # Check for stop loss hit
        if df.loc[prev_idx, 'trade_active'] and df.loc[curr_idx, 'Low'] <= df.loc[prev_idx, 'trailing_sl']:
            # Close position at stop loss
            exit_price = df.loc[prev_idx, 'trailing_sl']
            pnl = (exit_price - entry_price) * df.loc[prev_idx, 'position']
            df.loc[curr_idx, 'pnl'] = pnl
            df.loc[curr_idx, 'capital'] += pnl
            
            # Reset position
            df.loc[curr_idx, 'position'] = 0
            df.loc[curr_idx, 'trade_active'] = False
            df.loc[curr_idx, 'tp'] = np.nan
            df.loc[curr_idx, 'sl'] = np.nan
            df.loc[curr_idx, 'trailing_sl'] = np.nan
            df.loc[curr_idx, 'half_tp'] = np.nan
            df.loc[curr_idx, 'quarter_tp'] = np.nan
            df.loc[curr_idx, 'scaled_in'] = 0
            current_position = 0
        
        # Check for take profit hit
        elif df.loc[prev_idx, 'trade_active'] and df.loc[curr_idx, 'High'] >= df.loc[prev_idx, 'tp']:
            # Close position at take profit
            exit_price = df.loc[prev_idx, 'tp']
            pnl = (exit_price - entry_price) * df.loc[prev_idx, 'position']
            df.loc[curr_idx, 'pnl'] = pnl
            df.loc[curr_idx, 'capital'] += pnl
            
            # Reset position
            df.loc[curr_idx, 'position'] = 0
            df.loc[curr_idx, 'trade_active'] = False
            df.loc[curr_idx, 'tp'] = np.nan
            df.loc[curr_idx, 'sl'] = np.nan
            df.loc[curr_idx, 'trailing_sl'] = np.nan
            df.loc[curr_idx, 'half_tp'] = np.nan
            df.loc[curr_idx, 'quarter_tp'] = np.nan
            df.loc[curr_idx, 'scaled_in'] = 0
            current_position = 0
        
        # Check for scaling in at 50% of target (only if not yet scaled in at first level)
        elif (df.loc[prev_idx, 'trade_active'] and df.loc[prev_idx, 'scaled_in'] == 0 and 
              df.loc[curr_idx, 'High'] >= df.loc[prev_idx, 'half_tp']):
            # Add half the position size
            scale_in_size = initial_position_size / 2
            df.loc[curr_idx, 'position'] += scale_in_size
            current_position += scale_in_size
            df.loc[curr_idx, 'scaled_in'] = 1
        
        # Check for scaling in at 25% of target (only if scaled in once already)
        elif (df.loc[prev_idx, 'trade_active'] and df.loc[prev_idx, 'scaled_in'] == 1 and 
              df.loc[curr_idx, 'High'] >= df.loc[prev_idx, 'quarter_tp']):
            # Add quarter of the position size
            scale_in_size = initial_position_size / 4
            df.loc[curr_idx, 'position'] += scale_in_size
            current_position += scale_in_size
            df.loc[curr_idx, 'scaled_in'] = 2
        
        # Check for new entry signal
        if df.loc[curr_idx, 'signal'] == 1 and not df.loc[prev_idx, 'trade_active'] and ema_confirmation:
            # Set up new trade
            entry_price = df.loc[curr_idx, 'Close']
            target_price = entry_price + (df.loc[curr_idx, 'atr'] * 2.5)  # Take profit: ATR * 2.5
            
            # Stop loss: Low of last 2 candles
            stop_loss = min(df.loc[df.index[i-1], 'Low'], df.loc[df.index[i-2], 'Low'])
            
            # Set position
            df.loc[curr_idx, 'position'] = initial_position_size
            current_position = initial_position_size
            df.loc[curr_idx, 'trade_active'] = True
            df.loc[curr_idx, 'tp'] = target_price
            df.loc[curr_idx, 'sl'] = stop_loss
            df.loc[curr_idx, 'trailing_sl'] = stop_loss
            
            # Set scaling levels
            df.loc[curr_idx, 'half_tp'] = entry_price + (df.loc[curr_idx, 'atr'] * 2.5 * 0.5)  # 50% of target
            df.loc[curr_idx, 'quarter_tp'] = entry_price + (df.loc[curr_idx, 'atr'] * 2.5 * 0.75)  # 75% of target (25% remaining)
            df.loc[curr_idx, 'scaled_in'] = 0
        
        # Update trailing stop loss if price moves favorably
        if df.loc[curr_idx, 'trade_active']:
            # Trailing stop loss = min(current SL, ATR*2.5*0.25)
            trailing_sl_increment = df.loc[curr_idx, 'atr'] * 2.5 * 0.25
            potential_new_sl = df.loc[curr_idx, 'Close'] - trailing_sl_increment
            
            # Only move stop loss up, never down
            if potential_new_sl > df.loc[curr_idx, 'trailing_sl']:
                df.loc[curr_idx, 'trailing_sl'] = potential_new_sl
    
    # Calculate cumulative performance metrics
    df['cumulative_pnl'] = pd.Series(df['pnl']).cumsum()
    df['drawdown'] = pd.Series(df['capital']).cummax() - df['capital']
    
    return df

def visualize_supertrend_strategy(df):
    """
    Visualize the Supertrend strategy with entry/exit points
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with backtesting results
    """
    plt.figure(figsize=(14, 10))
    
    # Plot price and Supertrend
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price')
    
    # Plot Supertrend line
    supertrend_up = df.copy()
    supertrend_up.loc[df['supertrend_direction'] == -1, 'supertrend'] = np.nan
    supertrend_down = df.copy()
    supertrend_down.loc[df['supertrend_direction'] == 1, 'supertrend'] = np.nan
    
    plt.plot(supertrend_up.index, supertrend_up['supertrend'], 'g-', label='Supertrend (Up)')
    plt.plot(supertrend_down.index, supertrend_down['supertrend'], 'r-', label='Supertrend (Down)')
    
    # Plot EMA of EMAs
    plt.plot(df.index, df['ema_ema_10'], 'b--', alpha=0.7, label='EMA of EMA (10)')
    plt.plot(df.index, df['ema_ema_20'], 'purple', alpha=0.7, label='EMA of EMA (20)')
    
    # Plot entry points
    entries = df[df['signal'] == 1]
    exits = df[(pd.Series(df['position']).shift(1) > 0) & (df['position'] == 0)]
    
    plt.scatter(entries.index, entries['Close'], marker='^', color='g', s=100, label='Entry')
    plt.scatter(exits.index, exits['Close'], marker='v', color='r', s=100, label='Exit')
    
    # Plot stop loss and take profit levels
    plt.plot(df.index, df['tp'], 'g--', alpha=0.5, label='Take Profit')
    plt.plot(df.index, df['trailing_sl'], 'r--', alpha=0.5, label='Trailing Stop Loss')
    
    plt.title('Supertrend Strategy with HL/2 Approach')
    plt.legend()
    plt.grid(True)
    
    # Plot position size
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['position'], 'b-', label='Position Size')
    plt.title('Position Size')
    plt.legend()
    plt.grid(True)
    
    # Plot capital curve
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['capital'], 'g-', label='Account Capital')
    plt.title('Account Capital')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_strategy(data, atr_period=14, multiplier=2.5, initial_position=16):
    """
    Run the complete Supertrend strategy from raw price data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with OHLC price data
    atr_period : int
        Period for ATR calculation
    multiplier : float
        Multiplier for Supertrend bands
    initial_position : int
        Initial position size
    
    Returns:
    --------
    pandas.DataFrame with complete strategy results
    """
    # Calculate Supertrend
    df = calculate_supertrend(data, atr_period=atr_period, multiplier=multiplier)
    
    # Calculate double EMAs for confirmation
    df = calculate_double_ema(df)
    
    # Run backtest
    results = backtest_supertrend_strategy(df, initial_position_size=initial_position)
    
    # Calculate performance metrics
    total_trades = (pd.Series(results['signal']) == 1).sum()
    profitable_trades = ((pd.Series(results['pnl']) > 0) & (pd.Series(results['pnl']) != 0)).sum()
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    max_drawdown = pd.Series(results['drawdown']).max()
    final_capital = results['capital'].iloc[-1]
    roi = (final_capital / 100000 - 1) * 100
    
    print(f"Strategy Performance Summary:")
    print(f"----------------------------")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {profitable_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Max Drawdown: ${max_drawdown:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Return on Investment: {roi:.2%}")
    
    return results

# Example usage with sample data
if __name__ == "__main__":
    # Generate sample data for demonstration
    # np.random.seed(42)
    # dates = pd.date_range(start='2023-01-01', periods=200)
    # close_prices = np.random.normal(100, 1, 200).cumsum() + 500
    
    # data = pd.DataFrame({
    #     'Date': dates,
    #     'Open': close_prices - np.random.normal(0, 2, 200),
    #     'High': close_prices + np.random.normal(2, 1, 200),
    #     'Low': close_prices - np.random.normal(2, 1, 200),
    #     'Close': close_prices
    # })
    data = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\qts\features.csv")
    data.set_index('datetime', inplace=True)
    data.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
    
    # Run strategy
    results = run_strategy(data, initial_position=16)
    
    # Visualize results
    visualize_supertrend_strategy(results)
