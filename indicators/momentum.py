import pandas as pd
import numpy as np

class MomentumIndicator:
    def __init__(self):
        pass
    
    def rsi(self, df, length=14, price='close'):
        """
        Relative Strength Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for RSI calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: RSI values
        """
        # Calculate price changes
        delta = df[price].diff()
        
        # Create two series: one for gains, one for losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stoch_rsi(self, df, length=14, k_length=3, d_length=3, price='close'):
        """
        Stochastic RSI
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for RSI calculation
        k_length : int
            The window length for %K smoothing
        d_length : int
            The window length for %D smoothing
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with K and D values
        """
        # Calculate RSI
        rsi_values = self.rsi(df, length, price)
        
        # Calculate Stochastic RSI
        stoch_rsi = pd.DataFrame(index=df.index)
        
        # Min and max of RSI over the period
        min_rsi = rsi_values.rolling(window=length).min()
        max_rsi = rsi_values.rolling(window=length).max()
        
        # Calculate raw K (min-max normalization of RSI)
        stoch_rsi['K'] = 100 * ((rsi_values - min_rsi) / (max_rsi - min_rsi))
        
        # Apply smoothing to K
        stoch_rsi['K'] = stoch_rsi['K'].rolling(window=k_length).mean()
        
        # Calculate D (smoothed K)
        stoch_rsi['D'] = stoch_rsi['K'].rolling(window=d_length).mean()
        
        return stoch_rsi
    
    def stoch(self, df, k_length=14, k_smooth=1, d_length=3):
        """
        Stochastic Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        k_length : int
            The window length for %K calculation
        k_smooth : int
            The window length for %K smoothing
        d_length : int
            The window length for %D smoothing
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with K and D values
        """
        # Calculate min and max in the lookback period
        lowest_low = df['low'].rolling(window=k_length).min()
        highest_high = df['high'].rolling(window=k_length).max()
        
        # Calculate %K
        raw_k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Apply smoothing to %K if requested
        k = raw_k.rolling(window=k_smooth).mean() if k_smooth > 1 else raw_k
        
        # Calculate %D
        d = k.rolling(window=d_length).mean()
        
        result = pd.DataFrame({
            'K': k,
            'D': d
        }, index=df.index)
        
        return result
    
    def roc(self, df, length=10, price='close'):
        """
        Rate of Change
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for ROC calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: ROC values
        """
        # Calculate the rate of change
        roc = 100 * ((df[price] - df[price].shift(length)) / df[price].shift(length))
        
        return roc
    
    def ao(self, df, fast_length=5, slow_length=34):
        """
        Awesome Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        fast_length : int
            The window length for fast SMA
        slow_length : int
            The window length for slow SMA
            
        Returns:
        --------
        pandas.Series: AO values
        """
        # Calculate median price
        median_price = (df['high'] + df['low']) / 2
        
        # Calculate simple moving averages
        fast_sma = median_price.rolling(window=fast_length).mean()
        slow_sma = median_price.rolling(window=slow_length).mean()
        
        # Calculate AO
        ao = fast_sma - slow_sma
        
        return ao
    
    def cci(self, df, length=20):
        """
        Commodity Channel Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for CCI calculation
            
        Returns:
        --------
        pandas.Series: CCI values
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate moving average of typical price
        ma_tp = typical_price.rolling(window=length).mean()
        
        # Calculate mean deviation
        mean_deviation = typical_price.rolling(window=length).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - ma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def mom(self, df, length=10, price='close'):
        """
        Momentum
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for momentum calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: Momentum values
        """
        # Calculate momentum
        mom = df[price] - df[price].shift(length)
        
        return mom
    
    def williamsr(self, df, length=14):
        """
        Williams %R
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for Williams %R calculation
            
        Returns:
        --------
        pandas.Series: Williams %R values
        """
        # Calculate highest high and lowest low in the window
        highest_high = df['high'].rolling(window=length).max()
        lowest_low = df['low'].rolling(window=length).min()
        
        # Calculate Williams %R
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return williams_r
    
    def cmo(self, df, length=14, price='close'):
        """
        Chande Momentum Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for CMO calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: CMO values
        """
        # Calculate price changes
        delta = df[price].diff()
        
        # Create two series: one for gains, one for losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate sum of gains and losses over period
        sum_gain = gain.rolling(window=length).sum()
        sum_loss = loss.rolling(window=length).sum()
        
        # Calculate CMO
        cmo = 100 * ((sum_gain - sum_loss) / (sum_gain + sum_loss))
        
        return cmo
    
    def uo(self, df, short_length=7, medium_length=14, long_length=28, short_weight=4.0, medium_weight=2.0, long_weight=1.0):
        """
        Ultimate Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        short_length : int
            The short window length
        medium_length : int
            The medium window length
        long_length : int
            The long window length
        short_weight : float
            The weight for short average
        medium_weight : float
            The weight for medium average
        long_weight : float
            The weight for long average
            
        Returns:
        --------
        pandas.Series: Ultimate Oscillator values
        """
        # Calculate buying pressure (close - min(low, prior close))
        prior_close = df['close'].shift(1)
        min_low_prior = pd.concat([df['low'], prior_close], axis=1).min(axis=1)
        buying_pressure = df['close'] - min_low_prior
        
        # Calculate true range
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - prior_close),
            abs(df['low'] - prior_close)
        ], axis=1).max(axis=1)
        
        # Calculate averages
        avg_short = buying_pressure.rolling(window=short_length).sum() / tr.rolling(window=short_length).sum()
        avg_medium = buying_pressure.rolling(window=medium_length).sum() / tr.rolling(window=medium_length).sum()
        avg_long = buying_pressure.rolling(window=long_length).sum() / tr.rolling(window=long_length).sum()
        
        # Calculate UO
        total_weight = short_weight + medium_weight + long_weight
        uo = 100 * ((short_weight * avg_short + medium_weight * avg_medium + long_weight * avg_long) / total_weight)
        
        return uo
    
    def rsi_divergence(self, df, length=14, price='close'):
        """
        RSI Divergence
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for RSI calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with price, RSI, and divergence values
        """
        # Calculate RSI
        rsi_values = self.rsi(df, length, price)
        
        # Prepare result DataFrame
        result = pd.DataFrame(index=df.index)
        result['price'] = df[price]
        result['rsi'] = rsi_values
        
        # Calculate price momentum (slope)
        result['price_momentum'] = result['price'].diff(2)
        
        # Calculate RSI momentum (slope)
        result['rsi_momentum'] = result['rsi'].diff(2)
        
        # Simple divergence detection
        result['regular_bullish'] = ((result['price_momentum'] < 0) & (result['rsi_momentum'] > 0)).astype(int)
        result['regular_bearish'] = ((result['price_momentum'] > 0) & (result['rsi_momentum'] < 0)).astype(int)
        
        return result
    
    def kst(self, df, roc1_length=10, roc2_length=15, roc3_length=20, roc4_length=30, 
            smooth1_length=10, smooth2_length=10, smooth3_length=10, smooth4_length=15, 
            signal_length=9, price='close'):
        """
        Know Sure Thing (KST)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        roc*_length : int
            The window lengths for ROC calculations
        smooth*_length : int
            The window lengths for smoothing ROC values
        signal_length : int
            The window length for signal line calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with KST and signal values
        """
        # Calculate ROC values
        roc1 = self.roc(df, roc1_length, price).rolling(window=smooth1_length).mean()
        roc2 = self.roc(df, roc2_length, price).rolling(window=smooth2_length).mean()
        roc3 = self.roc(df, roc3_length, price).rolling(window=smooth3_length).mean()
        roc4 = self.roc(df, roc4_length, price).rolling(window=smooth4_length).mean()
        
        # Calculate KST
        kst = (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)
        
        # Calculate Signal line
        signal = kst.rolling(window=signal_length).mean()
        
        result = pd.DataFrame({
            'kst': kst,
            'signal': signal
        }, index=df.index)
        
        return result
    
    def stc(self, df, macd_fast=23, macd_slow=50, kst_length=10, d_length=3, price='close'):
        """
        Schaff Trend Cycle
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        macd_fast : int
            The fast EMA for MACD calculation
        macd_slow : int
            The slow EMA for MACD calculation
        kst_length : int
            The window length for Stochastic calculation on MACD
        d_length : int
            The window length for %D smoothing
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: STC values
        """
        # Calculate MACD
        ema_fast = df[price].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df[price].ewm(span=macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        # Calculate Stochastic of MACD
        stoch_k = pd.Series(index=df.index)
        
        for i in range(kst_length, len(macd)):
            window = macd.iloc[i-kst_length:i]
            min_val = window.min()
            max_val = window.max()
            
            if max_val - min_val != 0:
                stoch_k.iloc[i] = 100 * (macd.iloc[i] - min_val) / (max_val - min_val)
            else:
                stoch_k.iloc[i] = 0
        
        # Apply second Stochastic to first one
        stoch_d = pd.Series(index=df.index)
        
        for i in range(kst_length, len(stoch_k)):
            window = stoch_k.iloc[i-kst_length:i]
            valid_window = window.dropna()
            
            if not valid_window.empty:
                min_val = valid_window.min()
                max_val = valid_window.max()
                
                if max_val - min_val != 0:
                    stoch_d.iloc[i] = 100 * (stoch_k.iloc[i] - min_val) / (max_val - min_val)
                else:
                    stoch_d.iloc[i] = 0
        
        # Apply smoothing
        stc = stoch_d.rolling(window=d_length).mean()
        
        return stc
    
    def els(self, df, macd_fast=12, macd_slow=26, macd_signal=9, ema_length=13, price='close'):
        """
        Elder Impulse Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        macd_fast : int
            The fast EMA for MACD calculation
        macd_slow : int
            The slow EMA for MACD calculation
        macd_signal : int
            The signal line for MACD calculation
        ema_length : int
            The window length for EMA calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with impulse values (1: bullish, -1: bearish, 0: neutral)
        """
        # Calculate MACD
        ema_fast = df[price].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df[price].ewm(span=macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
        
        # Calculate MACD histogram
        macd_histogram = macd_line - macd_signal_line
        
        # Calculate EMA
        ema = df[price].ewm(span=ema_length, adjust=False).mean()
        
        # Calculate Elder Impulse
        result = pd.DataFrame(index=df.index)
        
        # MACD histogram color (1: green, -1: red, 0: same)
        result['macd_color'] = np.sign(macd_histogram.diff())
        
        # EMA color (1: green, -1: red, 0: same)
        result['ema_color'] = np.sign(ema.diff())
        
        # Elder Impulse System
        # Green bar: both MACD histogram and EMA rising
        # Red bar: both MACD histogram and EMA falling
        # Blue bar: mixed signals
        
        result['impulse'] = 0  # Initialize as neutral
        
        # Bullish impulse: both indicators rising
        result.loc[(result['macd_color'] > 0) & (result['ema_color'] > 0), 'impulse'] = 1
        
        # Bearish impulse: both indicators falling
        result.loc[(result['macd_color'] < 0) & (result['ema_color'] < 0), 'impulse'] = -1
        
        return result
    
    def squeeze_momentum_indicator(self, df, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
        """
        Squeeze Momentum Indicator (John Carter)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        bb_length : int
            The window length for Bollinger Bands calculation
        bb_mult : float
            The multiplier for Bollinger Bands
        kc_length : int
            The window length for Keltner Channel calculation
        kc_mult : float
            The multiplier for Keltner Channel
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with squeeze and momentum values
        """
        # Calculate Bollinger Bands
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=bb_length).mean()
        stdev = typical_price.rolling(window=bb_length).std(ddof=0)
        bb_upper = sma + (bb_mult * stdev)
        bb_lower = sma - (bb_mult * stdev)
        
        # Calculate Keltner Channel
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(window=kc_length).mean()
        
        kc_upper = sma + (kc_mult * atr)
        kc_lower = sma - (kc_mult * atr)
        
        # Is the market in a squeeze?
        squeeze = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
        
        # Calculate momentum
        highest_high = df['high'].rolling(window=bb_length).max()
        lowest_low = df['low'].rolling(window=bb_length).min()
        
        m1 = (highest_high + lowest_low) / 2
        m2 = (sma + sma) / 2
        
        momentum = typical_price - ((m1 + m2) / 2)
        momentum = momentum.rolling(window=1).mean()  # Optional smoothing
        
        result = pd.DataFrame({
            'squeeze': squeeze,
            'momentum': momentum
        }, index=df.index)
        
        return result
    
    def rvi(self, df, length=10):
        """
        Relative Vigor Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for RVI calculation
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with RVI and signal values
        """
        # Calculate numerator (close - open)
        co = df['close'] - df['open']
        
        # Calculate denominator (high - low)
        hl = df['high'] - df['low']
        
        # Calculate Vigor
        vigor = co / hl.where(hl != 0, 1)  # Avoid division by zero
        
        # Apply triangular weighting to Vigor over the period
        weights = np.arange(1, length + 1)
        sum_weights = np.sum(weights)
        
        # Calculate weighted RVI
        rvi = vigor.rolling(window=length).apply(
            lambda x: np.sum(weights * x) / sum_weights if len(x) == length else np.nan,
            raw=True
        )
        
        # Calculate signal line (3-period moving average)
        signal = rvi.rolling(window=4).mean()
        
        result = pd.DataFrame({
            'rvi': rvi,
            'signal': signal
        }, index=df.index)
        
        return result
    
    def dpo(self, df, length=20, price='close'):
        """
        Detrended Price Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for DPO calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: DPO values
        """
        # DPO = Price - Simple Moving Average (price, (length / 2) + 1) shifted (length / 2)
        shift_length = length // 2
        sma_length = shift_length + 1
        
        sma = df[price].rolling(window=sma_length).mean()
        dpo = df[price] - sma.shift(shift_length)
        
        return dpo
    
    def connors_rsi(self, df, rsi_length=3, streak_length=2, rank_length=100):
        """
        Connors RSI
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        rsi_length : int
            The window length for RSI calculation
        streak_length : int
            The window length for streak RSI calculation
        rank_length : int
            The window length for percentile rank calculation
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with RSI, streak RSI, percent rank, and Connors RSI values
        """
        result = pd.DataFrame(index=df.index)
        
        # Component 1: RSI of price changes
        result['rsi'] = self.rsi(df, rsi_length, 'close')
        
        # Component 2: RSI of streak
        # Calculate the streak
        price_change = df['close'].diff()
        streak = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:  # Up day
                if streak.iloc[i-1] > 0:
                    streak.iloc[i] = streak.iloc[i-1] + 1
                else:
                    streak.iloc[i] = 1
            elif price_change.iloc[i] < 0:  # Down day
                if streak.iloc[i-1] < 0:
                    streak.iloc[i] = streak.iloc[i-1] - 1
                else:
                    streak.iloc[i] = -1
        
        # Create a DataFrame for streak RSI calculation
        streak_df = pd.DataFrame({'close': streak}, index=df.index)
        result['streak_rsi'] = self.rsi(streak_df, streak_length, 'close')
        
        # Component 3: Percentile Rank
        result['percent_rank'] = df['close'].rolling(window=rank_length).apply(
            lambda x: self._percentile_rank(x[-1], x), 
            raw=True
        ) * 100
        
        # Calculate Connors RSI
        result['connors_rsi'] = (result['rsi'] + result['streak_rsi'] + result['percent_rank']) / 3
        
        return result
    
    def _percentile_rank(self, x, values):
        """
        Helper function to calculate percentile rank
        """
        if len(values) == 0:
            return np.nan
            
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            return np.nan
            
        rank = np.sum(values < x) / len(values)
        return rank
    
    def vortex(self, df, length=14):
        """
        Vortex Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for Vortex calculation
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with +VI and -VI values
        """
        # Calculate True Range
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close),
            'lc': abs(low - close)
        }).max(axis=1)
        
        # Calculate +VM and -VM
        vm_plus = abs(high - low.shift(1))
        vm_minus = abs(low - high.shift(1))
        
        # Calculate the sum over the specified length
        tr_sum = tr.rolling(window=length).sum()
        vm_plus_sum = vm_plus.rolling(window=length).sum()
        vm_minus_sum = vm_minus.rolling(window=length).sum()
        
        # Calculate +VI and -VI
        vi_plus = vm_plus_sum / tr_sum
        vi_minus = vm_minus_sum / tr_sum
        
        result = pd.DataFrame({
            '+VI': vi_plus,
            '-VI': vi_minus
        }, index=df.index)
        
        return result
    
    def choppiness(self, df, length=14):
        """
        Choppiness Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for Choppiness calculation
            
        Returns:
        --------
        pandas.Series: Choppiness values
        """
        # Calculate True Range
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close),
            'lc': abs(low - close)
        }).max(axis=1)
        
        # Calculate ATR sum over the period
        atr_sum = tr.rolling(window=length).sum()
        
        # Calculate the highest high and lowest low over the period
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        
        # Calculate Choppiness Index
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(length)
        
        return chop