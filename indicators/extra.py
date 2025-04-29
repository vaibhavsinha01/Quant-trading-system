import pandas as pd
import numpy as np
from scipy import stats


class ExtraIndicators:
    def __init__(self):
        pass
    
    def supertrend(self, df, atr_period=10, multiplier=3, high_col='high', low_col='low', close_col='close'):
        """
        SuperTrend Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        atr_period : int, default 10
            Period for ATR calculation
        multiplier : float, default 3
            Multiplier for ATR to determine band width
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        close_col : str, default 'close'
            Column name for close price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with SuperTrend indicator columns
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate ATR
        result['tr1'] = abs(result[high_col] - result[low_col])
        result['tr2'] = abs(result[high_col] - result[close_col].shift())
        result['tr3'] = abs(result[low_col] - result[close_col].shift())
        result['tr'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
        result['atr'] = result['tr'].rolling(window=atr_period).mean()
        
        # Calculate Basic Upper and Lower Bands
        hl2 = (result[high_col] + result[low_col]) / 2
        result['basic_upper_band'] = hl2 + (multiplier * result['atr'])
        result['basic_lower_band'] = hl2 - (multiplier * result['atr'])
        
        # Initialize Final Upper and Lower Bands columns
        result['final_upper_band'] = 0.0
        result['final_lower_band'] = 0.0
        
        # Calculate Final Upper and Lower Bands
        for i in range(atr_period, len(result)):
            # Final Upper Band
            if (result[close_col].iloc[i-1] <= result['final_upper_band'].iloc[i-1] or 
                result['basic_upper_band'].iloc[i] < result['basic_upper_band'].iloc[i-1]):
                result['final_upper_band'].iloc[i] = min(result['basic_upper_band'].iloc[i], result['final_upper_band'].iloc[i-1])
            else:
                result['final_upper_band'].iloc[i] = result['basic_upper_band'].iloc[i]
            
            # Final Lower Band
            if (result[close_col].iloc[i-1] >= result['final_lower_band'].iloc[i-1] or 
                result['basic_lower_band'].iloc[i] > result['basic_lower_band'].iloc[i-1]):
                result['final_lower_band'].iloc[i] = max(result['basic_lower_band'].iloc[i], result['final_lower_band'].iloc[i-1])
            else:
                result['final_lower_band'].iloc[i] = result['basic_lower_band'].iloc[i]
        
        # Initialize SuperTrend column
        result['supertrend'] = 0.0
        result['supertrend_direction'] = 0  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend and Direction
        for i in range(atr_period, len(result)):
            if result[close_col].iloc[i] <= result['final_upper_band'].iloc[i]:
                result['supertrend'].iloc[i] = result['final_upper_band'].iloc[i]
                result['supertrend_direction'].iloc[i] = -1
            else:
                result['supertrend'].iloc[i] = result['final_lower_band'].iloc[i]
                result['supertrend_direction'].iloc[i] = 1
        
        # Clean up interim columns
        result.drop(['tr1', 'tr2', 'tr3', 'tr', 'atr', 'basic_upper_band', 'basic_lower_band'], axis=1, inplace=True)
        
        return result
    
    def vervoort_tsi(self, df, long_period=25, short_period=13, signal_period=7, price_col='close'):
        """
        Vervoort variant of the True Strength Index (TSI)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        long_period : int, default 25
            Long period for EMA calculation
        short_period : int, default 13
            Short period for EMA calculation
        signal_period : int, default 7
            Signal line period
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Vervoort TSI indicator columns
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate momentum (price change)
        result['momentum'] = result[price_col].diff()
        
        # Double smoothing of momentum
        result['abs_momentum'] = abs(result['momentum'])
        result['momentum_ema1'] = result['momentum'].ewm(span=long_period, adjust=False).mean()
        result['abs_momentum_ema1'] = result['abs_momentum'].ewm(span=long_period, adjust=False).mean()
        
        result['momentum_ema2'] = result['momentum_ema1'].ewm(span=short_period, adjust=False).mean()
        result['abs_momentum_ema2'] = result['abs_momentum_ema1'].ewm(span=short_period, adjust=False).mean()
        
        # Calculate TSI
        result['tsi'] = 100 * (result['momentum_ema2'] / result['abs_momentum_ema2'])
        
        # Calculate signal line
        result['tsi_signal'] = result['tsi'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        result['tsi_histogram'] = result['tsi'] - result['tsi_signal']
        
        # Clean up interim columns
        result.drop(['momentum', 'abs_momentum', 'momentum_ema1', 'abs_momentum_ema1', 
                     'momentum_ema2', 'abs_momentum_ema2'], axis=1, inplace=True)
        
        return result
    
    def frama(self, df, window=16, fc=1, sc=200, price_col='close'):
        """
        Fractal Adaptive Moving Average (FRAMA)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 16
            Window size for calculation
        fc : int, default 1
            Fast constant - lowest value for alpha
        sc : int, default 200
            Slow constant - highest value for alpha
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with FRAMA column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate the Hurst exponent
        result['frama'] = np.nan
        
        # Set initial FRAMA value
        half_length = window // 2
        result['frama'].iloc[window-1] = result[price_col].iloc[:window].mean()
        
        for i in range(window, len(result)):
            # Calculate the dimension for the first half of the window
            price_max1 = result[price_col].iloc[i-window:i-half_length].max()
            price_min1 = result[price_col].iloc[i-window:i-half_length].min()
            n1 = (price_max1 - price_min1) / half_length if half_length > 0 else 0
            
            # Calculate the dimension for the second half of the window
            price_max2 = result[price_col].iloc[i-half_length:i].max()
            price_min2 = result[price_col].iloc[i-half_length:i].min()
            n2 = (price_max2 - price_min2) / half_length if half_length > 0 else 0
            
            # Calculate the dimension for the whole window
            price_max = result[price_col].iloc[i-window:i].max()
            price_min = result[price_col].iloc[i-window:i].min()
            n3 = (price_max - price_min) / window if window > 0 else 0
            
            # Avoid division by zero
            if n1 + n2 > 0:
                # Calculate the fractal dimension
                dimension = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            else:
                dimension = 1
            
            # Ensure dimension is between 1 and 2
            dimension = max(1, min(dimension, 2))
            
            # Calculate alpha
            alpha = np.exp(-4.6 * (dimension - 1))
            
            # Scale alpha between fc and sc
            alpha = alpha * (fc - sc) + sc
            alpha = 2.0 / (alpha + 1)
            
            # Calculate FRAMA
            result['frama'].iloc[i] = alpha * result[price_col].iloc[i] + (1 - alpha) * result['frama'].iloc[i-1]
        
        return result
    
    def regression_channel(self, df, window=20, deviations=2, price_col='close'):
        """
        Linear Regression Channel
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 20
            Window for linear regression calculation
        deviations : float, default 2
            Number of standard deviations for upper and lower bands
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Linear Regression Channel columns
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Add column for index to perform linear regression
        result['reg_middle'] = np.nan
        result['reg_upper'] = np.nan
        result['reg_lower'] = np.nan
        result['reg_slope'] = np.nan
        result['reg_intercept'] = np.nan
        result['reg_r2'] = np.nan
        
        for i in range(window, len(result) + 1):
            # Get the subset for the current window
            subset = result.iloc[i-window:i]
            
            # Create x values (time indices)
            x = np.arange(window)
            y = subset[price_col].values
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Store regression results
            result['reg_slope'].iloc[i-1] = slope
            result['reg_intercept'].iloc[i-1] = intercept
            result['reg_r2'].iloc[i-1] = r_value ** 2
            
            # Calculate the linear regression line
            y_fit = slope * (window - 1) + intercept  # The last point in the regression line
            result['reg_middle'].iloc[i-1] = y_fit
            
            # Calculate standard error of the estimation
            y_pred = slope * x + intercept
            std_error = np.sqrt(np.sum((y - y_pred) ** 2) / (window - 2))
            
            # Calculate upper and lower bands
            result['reg_upper'].iloc[i-1] = y_fit + deviations * std_error
            result['reg_lower'].iloc[i-1] = y_fit - deviations * std_error
        
        return result
    
    def vidya(self, df, short_period=9, long_period=26, price_col='close'):
        """
        Variable Index Dynamic Average (VIDYA)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        short_period : int, default 9
            Period for short CMO calculation
        long_period : int, default 26
            Period for long CMO calculation
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with VIDYA column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate price change
        result['price_change'] = result[price_col].diff()
        
        # Calculate Chande Momentum Oscillator (CMO)
        result['up_sum'] = np.where(result['price_change'] >= 0, result['price_change'], 0)
        result['down_sum'] = np.where(result['price_change'] < 0, -result['price_change'], 0)
        
        result['up_sum_short'] = result['up_sum'].rolling(window=short_period).sum()
        result['down_sum_short'] = result['down_sum'].rolling(window=short_period).sum()
        
        # Calculate short CMO
        result['cmo_short'] = 100 * (result['up_sum_short'] - result['down_sum_short']) / (result['up_sum_short'] + result['down_sum_short'])
        result['cmo_short'] = result['cmo_short'].fillna(0)
        result['cmo_short_abs'] = abs(result['cmo_short']) / 100
        
        # Calculate long CMO
        result['up_sum_long'] = result['up_sum'].rolling(window=long_period).sum()
        result['down_sum_long'] = result['down_sum'].rolling(window=long_period).sum()
        result['cmo_long'] = 100 * (result['up_sum_long'] - result['down_sum_long']) / (result['up_sum_long'] + result['down_sum_long'])
        result['cmo_long'] = result['cmo_long'].fillna(0)
        result['cmo_long_abs'] = abs(result['cmo_long']) / 100
        
        # Calculate VIDYA using CMO as volatility index
        # Initialize VIDYA with first price value
        result['vidya_short'] = np.nan
        result['vidya_long'] = np.nan
        
        # Set initial values
        result['vidya_short'].iloc[short_period] = result[price_col].iloc[short_period]
        result['vidya_long'].iloc[long_period] = result[price_col].iloc[long_period]
        
        # Calculate short-term VIDYA
        k_short = 2 / (short_period + 1)  # EMA alpha factor
        for i in range(short_period + 1, len(result)):
            alpha = k_short * result['cmo_short_abs'].iloc[i]
            result['vidya_short'].iloc[i] = alpha * result[price_col].iloc[i] + (1 - alpha) * result['vidya_short'].iloc[i-1]
        
        # Calculate long-term VIDYA
        k_long = 2 / (long_period + 1)  # EMA alpha factor
        for i in range(long_period + 1, len(result)):
            alpha = k_long * result['cmo_long_abs'].iloc[i]
            result['vidya_long'].iloc[i] = alpha * result[price_col].iloc[i] + (1 - alpha) * result['vidya_long'].iloc[i-1]
        
        # Clean up interim columns
        cols_to_drop = ['price_change', 'up_sum', 'down_sum', 'up_sum_short', 'down_sum_short', 
                         'cmo_short', 'cmo_short_abs', 'up_sum_long', 'down_sum_long', 'cmo_long', 'cmo_long_abs']
        result.drop(cols_to_drop, axis=1, inplace=True)
        
        return result
    
    def instantaneous_trendline(self, df, alpha=0.07, price_col='close'):
        """
        Instantaneous Trendline (IT)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        alpha : float, default 0.07
            Alpha parameter for sensitivity
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Instantaneous Trendline column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Initialize the IT column
        result['it'] = np.nan
        
        # Set initial IT value to first price
        result['it'].iloc[0] = result[price_col].iloc[0]
        
        for i in range(1, len(result)):
            # Calculate Instantaneous Trendline
            it_prev = result['it'].iloc[i-1]
            price = result[price_col].iloc[i]
            
            # Adjust alpha based on price direction
            if price > it_prev:
                alpha_adjusted = alpha
            else:
                alpha_adjusted = alpha * 2  # More responsive on downside
            
            # Calculate new IT
            result['it'].iloc[i] = (alpha_adjusted * price) + ((1 - alpha_adjusted) * it_prev)
        
        return result
    
    def zlsma(self, df, window=50, price_col='close'):
        """
        Zero Lag Smoothed Moving Average (ZLSMA)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        window : int, default 50
            Window for moving average calculation
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ZLSMA column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Calculate the lag period (usually a percentage of the window)
        lag = (window - 1) // 2
        
        # Calculate the "error correction" term
        result['error_correction'] = result[price_col] + (result[price_col] - result[price_col].shift(lag))
        
        # Calculate the Zero Lag EMA
        result['zlsma'] = result['error_correction'].ewm(span=window, adjust=False).mean()
        
        # Clean up interim columns
        result.drop(['error_correction'], axis=1, inplace=True)
        
        return result
    
    def mcginley_dynamic(self, df, period=14, k=0.6, price_col='close'):
        """
        McGinley Dynamic Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        period : int, default 14
            Period for moving average calculation
        k : float, default 0.6
            Sensitivity constant
        price_col : str, default 'close'
            Column name for price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with McGinley Dynamic column
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Initialize McGinley Dynamic column
        result['mcginley_dynamic'] = np.nan
        
        # Set first value to a simple moving average
        result['mcginley_dynamic'].iloc[period-1] = result[price_col].iloc[:period].mean()
        
        # Calculate McGinley Dynamic
        for i in range(period, len(result)):
            md_prev = result['mcginley_dynamic'].iloc[i-1]
            price = result[price_col].iloc[i]
            
            # McGinley Dynamic formula
            md = md_prev + ((price - md_prev) / (k * period * pow(price / md_prev, 4)))
            result['mcginley_dynamic'].iloc[i] = md
        
        return result
    
    def ib_boxes(self, df, high_col='high', low_col='low', open_col='open', close_col='close', period=None):
        """
        Inside Bar Boxes - Identifies inside bars (price action entirely within previous bar's range)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        high_col : str, default 'high'
            Column name for high price data
        low_col : str, default 'low'
            Column name for low price data
        open_col : str, default 'open'
            Column name for open price data
        close_col : str, default 'close'
            Column name for close price data
        period : int, default None
            Optional period to look back for inside bars. If None, checks only immediate predecessor
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Inside Bar identification columns
        """
        # Make a copy of the dataframe
        result = df.copy()
        
        # Initialize inside bar column (1 for inside bar, 0 for not)
        result['inside_bar'] = 0
        
        # Initialize columns for box boundaries
        result['ib_high'] = np.nan
        result['ib_low'] = np.nan
        
        if period is None:
            # Check for classic inside bars (current bar within previous bar)
            for i in range(1, len(result)):
                prev_high = result[high_col].iloc[i-1]
                prev_low = result[low_col].iloc[i-1]
                curr_high = result[high_col].iloc[i]
                curr_low = result[low_col].iloc[i]
                
                # Check if current bar is contained within previous bar
                if curr_high <= prev_high and curr_low >= prev_low:
                    result['inside_bar'].iloc[i] = 1
                    result['ib_high'].iloc[i] = prev_high
                    result['ib_low'].iloc[i] = prev_low
        else:
            # Check for inside bars within the specified period
            for i in range(period, len(result)):
                period_high = result[high_col].iloc[i-period:i].max()
                period_low = result[low_col].iloc[i-period:i].min()
                curr_high = result[high_col].iloc[i]
                curr_low = result[low_col].iloc[i]
                
                # Check if current bar is contained within the range of the period
                if curr_high <= period_high and curr_low >= period_low:
                    result['inside_bar'].iloc[i] = 1
                    result['ib_high'].iloc[i] = period_high
                    result['ib_low'].iloc[i] = period_low
        
        # Identify breakouts from inside bars
        result['ib_breakout'] = 0  # 1 for upside breakout, -1 for downside breakout
        
        for i in range(1, len(result)):
            if result['inside_bar'].iloc[i-1] == 1:
                if result[close_col].iloc[i] > result['ib_high'].iloc[i-1]:
                    result['ib_breakout'].iloc[i] = 1  # Upside breakout
                elif result[close_col].iloc[i] < result['ib_low'].iloc[i-1]:
                    result['ib_breakout'].iloc[i] = -1  # Downside breakout
        
        # Forward fill the box boundaries for visualization
        result['ib_high_line'] = None
        result['ib_low_line'] = None
        
        ib_count = 0
        current_high = None
        current_low = None
        
        for i in range(len(result)):
            if result['inside_bar'].iloc[i] == 1:
                # Start or continue an inside bar sequence
                ib_count += 1
                if ib_count == 1:  # First inside bar in sequence
                    current_high = result['ib_high'].iloc[i]
                    current_low = result['ib_low'].iloc[i]
                
                result['ib_high_line'].iloc[i] = current_high
                result['ib_low_line'].iloc[i] = current_low
            elif result['ib_breakout'].iloc[i] != 0:
                # Reset on breakout
                ib_count = 0
                current_high = None
                current_low = None
            elif ib_count > 0:
                # Continue drawing the box until breakout
                result['ib_high_line'].iloc[i] = current_high
                result['ib_low_line'].iloc[i] = current_low
        
        return result