import pandas as pd
import numpy as np


class VolatilityIndicator:
    def __init__(self):
        pass
    
    def bbands(self, df, window=14, stdev_factor=2, price_col='close'):
        """
        Bollinger Bands
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with 'price_col' column
        window : int, default 20
            Window for moving average
        stdev_factor : float, default 2
            Number of standard deviations for upper and lower bands
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Bollinger Bands columns (middle, upper, lower, bandwidth, %b)
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate the middle band (SMA)
        result['bb_middle'] = result[price_col].rolling(window=window).mean()
        
        # Calculate standard deviation of price
        stdev = result[price_col].rolling(window=window).std()
        
        # Calculate upper and lower bands
        result['bb_upper'] = result['bb_middle'] + (stdev * stdev_factor)
        result['bb_lower'] = result['bb_middle'] - (stdev * stdev_factor)
        
        # Calculate bandwidth: (Upper - Lower) / Middle
        result['bb_bandwidth'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # Calculate %B: (Price - Lower) / (Upper - Lower)
        result['bb_percent_b'] = (result[price_col] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # return result
        return result['bb_middle'],result['bb_upper'],result['bb_lower'],result['bb_bandwidth'],result['bb_percent_b']
    
    def atr(self, df, window=14, high_col='high', low_col='low', close_col='close'):
        """
        Average True Range (ATR)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 14
            Window for ATR calculation
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        close_col : str, default 'close'
            Column name for close price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ATR column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate true range
        result['tr1'] = abs(result[high_col] - result[low_col])
        result['tr2'] = abs(result[high_col] - result[close_col].shift())
        result['tr3'] = abs(result[low_col] - result[close_col].shift())
        result['tr'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR using EMA
        result['atr'] = result['tr'].ewm(span=window, adjust=False).mean()
        
        # Clean up interim columns
        result.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
        
        # return result
        return result['atr']
    
    def volatility_stop(self, df, atr_window=14, multiplier=3, high_col='high', low_col='low', close_col='close'):
        """
        Volatility Stop (Chandelier Exit)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        atr_window : int, default 14
            Window for ATR calculation
        multiplier : float, default 3
            Multiplier for ATR
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        close_col : str, default 'close'
            Column name for close price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with long and short Chandelier Exit columns
        """
        # Calculate ATR
        result = self.atr(df, window=atr_window, high_col=high_col, low_col=low_col, close_col=close_col)
        
        # Calculate highest high and lowest low over the window
        result['highest_high'] = result[high_col].rolling(window=atr_window).max()
        result['lowest_low'] = result[low_col].rolling(window=atr_window).min()
        
        # Calculate Chandelier Exit Long (for short positions)
        result['chandelier_exit_long'] = result['highest_high'] - (multiplier * result['atr'])
        
        # Calculate Chandelier Exit Short (for long positions)
        result['chandelier_exit_short'] = result['lowest_low'] + (multiplier * result['atr'])
        
        # Clean up interim columns
        result.drop(['highest_high', 'lowest_low'], axis=1, inplace=True)
        
        return result
    
    def high_low_range_indicator(self, df, window=14, high_col='high', low_col='low'):
        """
        High-Low Range Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 14
            Window for calculation
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with High-Low Range Indicator column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate high-low range for each period
        result['hl_range'] = result[high_col] - result[low_col]
        
        # Calculate average high-low range over the window
        result['avg_hl_range'] = result['hl_range'].rolling(window=window).mean()
        
        # Calculate high-low range indicator (current range relative to average)
        result['hl_range_indicator'] = result['hl_range'] / result['avg_hl_range']
        
        # Clean up interim columns
        result.drop(['hl_range'], axis=1, inplace=True)
        
        return result
    
    def moving_average_envelope(self, df, window=20, envelope_pct=2.5, price_col='close'):
        """
        Moving Average Envelope
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 20
            Window for moving average calculation
        envelope_pct : float, default 2.5
            Percentage displacement for upper and lower bands
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Moving Average Envelope columns
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate the moving average
        result['ma_envelope_middle'] = result[price_col].rolling(window=window).mean()
        
        # Calculate the envelope percentage as a decimal
        envelope_decimal = envelope_pct / 100.0
        
        # Calculate upper and lower bands
        result['ma_envelope_upper'] = result['ma_envelope_middle'] * (1 + envelope_decimal)
        result['ma_envelope_lower'] = result['ma_envelope_middle'] * (1 - envelope_decimal)
        
        return result
    
    def ulcer_index(self, df, window=14, price_col='close'):
        """
        Ulcer Index - measures downside risk
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 14
            Window for calculation
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Ulcer Index column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate the highest price in the lookback period
        result['highest_high'] = result[price_col].rolling(window=window).max()
        
        # Calculate percentage drawdown from highest price
        result['pct_drawdown'] = (result[price_col] / result['highest_high'] - 1) * 100
        
        # Square the drawdowns (only consider negative values, set positive to zero)
        result['squared_drawdown'] = result['pct_drawdown'].apply(lambda x: x**2 if x < 0 else 0)
        
        # Calculate the mean of squared drawdowns over the window
        result['mean_squared_drawdown'] = result['squared_drawdown'].rolling(window=window).mean()
        
        # Calculate Ulcer Index (square root of mean squared drawdown)
        result['ulcer_index'] = np.sqrt(result['mean_squared_drawdown'])
        
        # Clean up interim columns
        result.drop(['highest_high', 'pct_drawdown', 'squared_drawdown', 'mean_squared_drawdown'], axis=1, inplace=True)
        
        return result
    
    def normalized_atr(self, df, window=14, high_col='high', low_col='low', close_col='close', price_col='close'):
        """
        Normalized Average True Range (ATR relative to price)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 14
            Window for ATR calculation
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        close_col : str, default 'close'
            Column name for close price data
        price_col : str, default 'close'
            Column name for price to normalize against
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Normalized ATR column
        """
        # Calculate ATR
        result = self.atr(df, window=window, high_col=high_col, low_col=low_col, close_col=close_col)
        
        # Normalize ATR to price (as percentage)
        result['norm_atr'] = (result['atr'] / result[price_col]) * 100
        
        return result
    
    def relative_volatility_index(self, df, window=10, rvi_window=14, price_col='close'):
        """
        Relative Volatility Index (RVI)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 10
            Window for standard deviation calculation
        rvi_window : int, default 14
            Window for RVI calculation
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with RVI column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate standard deviation over the window
        result['stdev'] = result[price_col].rolling(window=window).std()
        
        # Calculate price change
        result['price_change'] = result[price_col].diff()
        
        # Define upward and downward volatility
        result['up_vol'] = np.where(result['price_change'] >= 0, result['stdev'], 0)
        result['down_vol'] = np.where(result['price_change'] < 0, result['stdev'], 0)
        
        # Calculate smoothed up and down volatility
        result['smoothed_up_vol'] = result['up_vol'].ewm(span=rvi_window, adjust=False).mean()
        result['smoothed_down_vol'] = result['down_vol'].ewm(span=rvi_window, adjust=False).mean()
        
        # Calculate RVI: smoothed_up_vol / (smoothed_up_vol + smoothed_down_vol)
        result['rvi'] = 100 * (result['smoothed_up_vol'] / 
                            (result['smoothed_up_vol'] + result['smoothed_down_vol']))
        
        # Clean up interim columns
        result.drop(['stdev', 'price_change', 'up_vol', 'down_vol', 
                     'smoothed_up_vol', 'smoothed_down_vol'], axis=1, inplace=True)
        
        return result
    
    def cboe_vix(self, df, short_window=9, long_window=30, price_col='close'):
        """
        A simplified VIX-like calculation (not the actual CBOE VIX methodology)
        This is an approximation since the real VIX uses options prices
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        short_window : int, default 9
            Window for short-term volatility
        long_window : int, default 30
            Window for long-term volatility
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with simplified VIX column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate log returns
        result['log_return'] = np.log(result[price_col] / result[price_col].shift(1))
        
        # Calculate rolling standard deviation of returns
        result['short_std'] = result['log_return'].rolling(window=short_window).std()
        result['long_std'] = result['log_return'].rolling(window=long_window).std()
        
        # Annualize volatility (multiply by sqrt of 252 trading days)
        result['simplified_vix'] = result['short_std'] * np.sqrt(252) * 100
        result['long_vix'] = result['long_std'] * np.sqrt(252) * 100
        
        # Calculate ratio of short to long volatility
        result['vix_ratio'] = result['simplified_vix'] / result['long_vix']
        
        # Clean up interim columns
        result.drop(['log_return', 'short_std', 'long_std'], axis=1, inplace=True)
        
        return result
    
    def z_score(self, df, window=20, price_col='close'):
        """
        Z-Score of price
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 20
            Window for calculation
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Z-Score column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate rolling mean
        result['rolling_mean'] = result[price_col].rolling(window=window).mean()
        
        # Calculate rolling standard deviation
        result['rolling_std'] = result[price_col].rolling(window=window).std()
        
        # Calculate z-score
        result['z_score'] = (result[price_col] - result['rolling_mean']) / result['rolling_std']
        
        # Clean up interim columns
        result.drop(['rolling_mean', 'rolling_std'], axis=1, inplace=True)
        
        return result
    
    def true_range_percent(self, df, high_col='high', low_col='low', close_col='close'):
        """
        True Range Percent - True Range normalized by close price
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        close_col : str, default 'close'
            Column name for close price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with True Range Percent column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate true range
        result['tr1'] = abs(result[high_col] - result[low_col])
        result['tr2'] = abs(result[high_col] - result[close_col].shift())
        result['tr3'] = abs(result[low_col] - result[close_col].shift())
        result['tr'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate true range percent
        result['tr_percent'] = result['tr'] / result[close_col].shift() * 100
        
        # Clean up interim columns
        result.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
        
        return result
    
    def volatility_ratio(self, df, short_window=5, long_window=20, price_col='close'):
        """
        Volatility Ratio - Compares short-term volatility to long-term volatility
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        short_window : int, default 5
            Window for short-term volatility
        long_window : int, default 20
            Window for long-term volatility
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Volatility Ratio column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate returns
        result['returns'] = result[price_col].pct_change()
        
        # Calculate short-term and long-term standard deviation
        result['short_vol'] = result['returns'].rolling(window=short_window).std()
        result['long_vol'] = result['returns'].rolling(window=long_window).std()
        
        # Calculate volatility ratio
        result['vol_ratio'] = result['short_vol'] / result['long_vol']
        
        # Clean up interim columns
        result.drop(['returns', 'short_vol', 'long_vol'], axis=1, inplace=True)
        
        return result
    
    def fractal_chaos_bands(self, df, window=10, high_col='high', low_col='low'):
        """
        Fractal Chaos Bands - Identifies local maxima and minima
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 10
            Window for fractal identification
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Fractal Chaos Bands columns
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Initialize the fractal columns
        result['fractal_high'] = np.nan
        result['fractal_low'] = np.nan
        
        # Half window size (center point + sides)
        half_window = window // 2
        
        # Identify high fractals (bullish)
        for i in range(half_window, len(result) - half_window):
            # Get window of data
            window_data = result.iloc[i-half_window:i+half_window+1]
            
            # Check if center point is highest in window
            if window_data[high_col].idxmax() == result.index[i]:
                result.loc[result.index[i], 'fractal_high'] = result.loc[result.index[i], high_col]
        
        # Identify low fractals (bearish)
        for i in range(half_window, len(result) - half_window):
            # Get window of data
            window_data = result.iloc[i-half_window:i+half_window+1]
            
            # Check if center point is lowest in window
            if window_data[low_col].idxmin() == result.index[i]:
                result.loc[result.index[i], 'fractal_low'] = result.loc[result.index[i], low_col]
        
        # Forward fill the fractal values to create the bands
        result['fractal_high_band'] = result['fractal_high'].fillna(method='ffill')
        result['fractal_low_band'] = result['fractal_low'].fillna(method='ffill')
        
        # Drop intermediate columns
        result.drop(['fractal_high', 'fractal_low'], axis=1, inplace=True)
        
        return result
    
    def directional_volatility(self, df, window=14, high_col='high', low_col='low', close_col='close'):
        """
        Directional Volatility - Measures if volatility is trending up or down
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 14
            Window for calculation
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        close_col : str, default 'close'
            Column name for close price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Directional Volatility columns
        """
        # Calculate ATR first
        result = self.atr(df, window=window, high_col=high_col, low_col=low_col, close_col=close_col)
        
        # Calculate the rate of change of ATR
        result['atr_roc'] = result['atr'].pct_change(periods=window) * 100
        atr_roc = result['atr'].pct_change(periods=window) * 100
        
        # Calculate EMA of ATR Rate of Change
        result['atr_roc_ema'] = result['atr_roc'].ewm(span=window, adjust=False).mean()
        atr_roc_ema = result['atr_roc'].ewm(span=window, adjust=False).mean()
        
        # Create directional volatility indicator (positive = increasing volatility)
        result['directional_vol'] = np.where(result['atr_roc_ema'] > 0, 1, -1)
        directional_vol = np.where(result['atr_roc_ema'] > 0, 1, -1)
        
        # Volatility trend strength
        result['vol_trend_strength'] = abs(result['atr_roc_ema'])
        vol_trend_strength = abs(result['atr_roc_ema'])
        
        return atr_roc,atr_roc_ema,directional_vol,vol_trend_strength
        # return result