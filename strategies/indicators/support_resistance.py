import pandas as pd
import numpy as np


class SupportResistance:
    def __init__(self):
        """
        Initialize the Support and Resistance detection class.
        """
        pass

    def pivot_points(self, df, method='standard', period=1):
        """
        Calculate pivot points using different methods.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLC data. Must have columns: 'high', 'low', 'close'
        method : str, default 'standard'
            Method to calculate pivot points. Options:
            - 'standard': Standard pivot points
            - 'fibonacci': Fibonacci pivot points
            - 'camarilla': Camarilla pivot points
            - 'woodie': Woodie pivot points
        period : int, default 1
            Number of periods to look back for calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with pivot points and support/resistance levels
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()
        
        # Calculate previous period's data
        prev_high = df['high'].rolling(period).max().shift(1)
        prev_low = df['low'].rolling(period).min().shift(1)
        prev_close = df['close'].shift(1)
        
        # For some methods, we need the previous period's open
        if method in ['woodie']:
            prev_open = df['open'].shift(1)
        
        # Calculate pivot point based on selected method
        if method == 'standard':
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Support levels
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
            # Resistance levels
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            
        elif method == 'fibonacci':
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Fibonacci ratios
            s1 = pivot - 0.382 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            s3 = pivot - 1.000 * (prev_high - prev_low)
            
            r1 = pivot + 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.000 * (prev_high - prev_low)
            
        elif method == 'camarilla':
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Camarilla ratios
            s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
            s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
            s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
            
            r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
            r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
            r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
            
        elif method == 'woodie':
            pivot = (prev_high + prev_low + 2 * prev_close) / 4
            
            # Support levels
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
            # Resistance levels
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            
        else:
            raise ValueError(f"Method '{method}' not recognized. Please use 'standard', 'fibonacci', 'camarilla', or 'woodie'.")
        
        # Add pivot points to the result DataFrame
        result['pivot'] = pivot
        result['s1'] = s1
        result['s2'] = s2
        result['s3'] = s3
        result['r1'] = r1
        result['r2'] = r2
        result['r3'] = r3
        
        return result

    def swings_high_low(self, df, window=5, threshold=0):
        """
        Identify swing highs and lows in price data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLC data. Must have columns: 'high', 'low'
        window : int, default 5
            Number of periods to look before and after for swing identification
        threshold : float, default 0
            Minimum percentage difference required to identify a swing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional columns for swing highs and lows
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()
        
        # Initialize swing high/low columns
        result['swing_high'] = np.nan
        result['swing_low'] = np.nan
        
        # We need at least 2*window + 1 periods to identify a swing
        if len(df) < 2 * window + 1:
            return result
        
        # Loop through the DataFrame to identify swings
        for i in range(window, len(df) - window):
            # Extract the current window
            current_window = df.iloc[i - window:i + window + 1]
            
            # Check if current point is a swing high
            is_swing_high = True
            for j in range(1, window + 1):
                if (current_window['high'].iloc[window] <= current_window['high'].iloc[window - j] or
                    current_window['high'].iloc[window] <= current_window['high'].iloc[window + j]):
                    is_swing_high = False
                    break
                    
            # Check if current point is a swing low
            is_swing_low = True
            for j in range(1, window + 1):
                if (current_window['low'].iloc[window] >= current_window['low'].iloc[window - j] or
                    current_window['low'].iloc[window] >= current_window['low'].iloc[window + j]):
                    is_swing_low = False
                    break
            
            # Apply threshold filter if specified
            if threshold > 0:
                if is_swing_high:
                    # Calculate percentage difference from surrounding points
                    pct_diff = min(
                        [(current_window['high'].iloc[window] / current_window['high'].iloc[window - j] - 1) for j in range(1, window + 1)] +
                        [(current_window['high'].iloc[window] / current_window['high'].iloc[window + j] - 1) for j in range(1, window + 1)]
                    )
                    is_swing_high = pct_diff >= threshold
                    
                if is_swing_low:
                    # Calculate percentage difference from surrounding points
                    pct_diff = min(
                        [(current_window['low'].iloc[window - j] / current_window['low'].iloc[window] - 1) for j in range(1, window + 1)] +
                        [(current_window['low'].iloc[window + j] / current_window['low'].iloc[window] - 1) for j in range(1, window + 1)]
                    )
                    is_swing_low = pct_diff >= threshold
            
            # Assign swing values
            if is_swing_high:
                result.iloc[i, result.columns.get_loc('swing_high')] = df['high'].iloc[i]
            
            if is_swing_low:
                result.iloc[i, result.columns.get_loc('swing_low')] = df['low'].iloc[i]
        
        return result

    def order_block_zone(self, df, lookback=10, threshold=0.01, bullish=True, bearish=True):
        """
        Identify order block zones based on price reversals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLC data. Must have columns: 'open', 'high', 'low', 'close'
        lookback : int, default 10
            Number of periods to look back for order block identification
        threshold : float, default 0.01
            Minimum percentage price move to identify a reversal
        bullish : bool, default True
            Whether to identify bullish order blocks
        bearish : bool, default True
            Whether to identify bearish order blocks
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional columns for bullish and bearish order blocks
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()
        
        # Initialize order block columns
        if bullish:
            result['bullish_ob_top'] = np.nan
            result['bullish_ob_bottom'] = np.nan
        
        if bearish:
            result['bearish_ob_top'] = np.nan
            result['bearish_ob_bottom'] = np.nan
        
        # Calculate price change percentage
        result['price_change_pct'] = result['close'].pct_change()
        
        # Loop through the DataFrame to identify order blocks
        for i in range(lookback, len(df)):
            # Bullish order block (reversal from bearish to bullish)
            if bullish:
                # Look for a strong bullish move
                if result['price_change_pct'].iloc[i] > threshold:
                    # Look back for the most recent bearish candle
                    for j in range(i-1, max(0, i-lookback-1), -1):
                        if df['close'].iloc[j] < df['open'].iloc[j]:
                            # This is a bearish candle preceding the bullish move
                            result.loc[df.index[j], 'bullish_ob_top'] = df['high'].iloc[j]
                            result.loc[df.index[j], 'bullish_ob_bottom'] = df['low'].iloc[j]
                            break
            
            # Bearish order block (reversal from bullish to bearish)
            if bearish:
                # Look for a strong bearish move
                if result['price_change_pct'].iloc[i] < -threshold:
                    # Look back for the most recent bullish candle
                    for j in range(i-1, max(0, i-lookback-1), -1):
                        if df['close'].iloc[j] > df['open'].iloc[j]:
                            # This is a bullish candle preceding the bearish move
                            result.loc[df.index[j], 'bearish_ob_top'] = df['high'].iloc[j]
                            result.loc[df.index[j], 'bearish_ob_bottom'] = df['low'].iloc[j]
                            break
        
        # Drop the temporary column
        result.drop('price_change_pct', axis=1, inplace=True)
        
        return result

    def zigzag_peak_valley(self, df, pct_change=5, column='close'):
        """
        Implement zigzag indicator to identify significant peaks and valleys.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price data
        pct_change : float, default 5
            Minimum percentage change required to identify a zigzag point
        column : str, default 'close'
            Column name to use for zigzag calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional column for zigzag points
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()
        
        # Initialize zigzag column
        result['zigzag'] = np.nan
        
        # Track the current trend and last zigzag point
        prices = df[column].values
        last_zigzag_value = prices[0]
        result.loc[df.index[0], 'zigzag'] = last_zigzag_value
        
        # 1 for uptrend, -1 for downtrend
        trend = 0
        
        # Loop through prices to identify zigzag points
        for i in range(1, len(prices)):
            # Calculate percentage change from last zigzag point
            current_change = (prices[i] / last_zigzag_value - 1) * 100
            
            # Initialize trend if not set
            if trend == 0:
                trend = 1 if current_change > 0 else -1
            
            # Check if change exceeds threshold in the direction of the trend
            if (trend == 1 and current_change > pct_change) or (trend == -1 and current_change < -pct_change):
                # Record this point as a zigzag point
                result.loc[df.index[i], 'zigzag'] = prices[i]
                last_zigzag_value = prices[i]
            
            # Check if trend has reversed significantly
            if (trend == 1 and current_change < -pct_change) or (trend == -1 and current_change > pct_change):
                # Record this point as a zigzag point
                result.loc[df.index[i], 'zigzag'] = prices[i]
                last_zigzag_value = prices[i]
                # Reverse the trend
                trend = -trend
        
        return result

    def mean_reversion_channel(self, df, period=20, std_multiplier=2):
        """
        Calculate a mean reversion channel based on moving average and standard deviation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price data. Must have a 'close' column.
        period : int, default 20
            The look-back period for calculating the moving average
        std_multiplier : float, default 2
            The number of standard deviations to use for the channel width
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional columns for mean reversion channel boundaries
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()
        
        # Calculate the moving average
        result['ma'] = result['close'].rolling(window=period).mean()
        
        # Calculate the standard deviation
        result['std'] = result['close'].rolling(window=period).std()
        
        # Calculate upper and lower channel boundaries
        result['upper_band'] = result['ma'] + (std_multiplier * result['std'])
        result['lower_band'] = result['ma'] - (std_multiplier * result['std'])
        
        # Identify potential support and resistance zones
        result['support_zone'] = np.where(
            (result['close'] < result['lower_band']) & 
            (result['close'].shift(1) >= result['lower_band'].shift(1)),
            result['lower_band'],
            np.nan
        )
        
        result['resistance_zone'] = np.where(
            (result['close'] > result['upper_band']) & 
            (result['close'].shift(1) <= result['upper_band'].shift(1)),
            result['upper_band'],
            np.nan
        )
        
        # Cleanup - drop intermediate columns if needed
        # result.drop(['ma', 'std'], axis=1, inplace=True)
        
        return result

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'open': [100, 101, 103, 102, 105, 107, 109, 108, 110, 112],
        'high': [103, 104, 105, 106, 107, 110, 112, 111, 113, 115],
        'low': [99, 100, 101, 100, 103, 105, 107, 106, 108, 110],
        'close': [101, 103, 102, 105, 107, 109, 108, 110, 112, 114]
    }
    df = pd.DataFrame(data)
    
    # Initialize support resistance class
    sr = SupportResistance()
    
    # Test pivot points
    pivot_df = sr.pivot_points(df, method='standard')
    print("Pivot Points:")
    print(pivot_df[['close', 'pivot', 's1', 'r1']].tail())
    
    # Test swing high/low
    swing_df = sr.swings_high_low(df, window=2)
    print("\nSwing High/Low:")
    print(swing_df[['close', 'swing_high', 'swing_low']].tail())
    
    # Test order block zone
    ob_df = sr.order_block_zone(df, lookback=3)
    print("\nOrder Block Zone:")
    print(ob_df[['close', 'bullish_ob_top', 'bearish_ob_bottom']].tail())
    
    # Test zigzag
    zigzag_df = sr.zigzag_peak_valley(df, pct_change=1)
    print("\nZigzag:")
    print(zigzag_df[['close', 'zigzag']].tail())
    
    # Test mean reversion channel
    mrc_df = sr.mean_reversion_channel(df, period=5)
    print("\nMean Reversion Channel:")
    print(mrc_df[['close', 'ma', 'upper_band', 'lower_band']].tail())