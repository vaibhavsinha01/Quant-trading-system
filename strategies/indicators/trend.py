import pandas as pd
import numpy as np

# add more trend indicators
class TrendIndicators:
    def __init__(self):
        pass
    
    def ema(self, df, length=14, price='close'):
        """
        Exponential Moving Average
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for EMA calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: EMA values
        """
        ema = df[price].ewm(span=length, adjust=False).mean()
        return ema
    
    def dema(self, df, length=14, price='close'):
        """
        Double Exponential Moving Average
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for DEMA calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: DEMA values
        """
        ema1 = self.ema(df, length, price)
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        dema = (2 * ema1) - ema2
        return dema
    
    def tema(self, df, length=14, price='close'):
        """
        Triple Exponential Moving Average
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for TEMA calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: TEMA values
        """
        ema1 = self.ema(df, length, price)
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema
    
    def t3ribbon(self, df, lengths=[5, 8, 13, 21, 34], price='close'):
        """
        T3 Ribbon (multiple T3 moving averages)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        lengths : list
            List of window lengths for T3 calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with T3 values for each length
        """
        result = pd.DataFrame(index=df.index)
        
        for length in lengths:
            # Calculate T3 (Tillson T3)
            # Volume factor (typical values 0.7-0.8)
            vfactor = 0.7
            
            # Calculate multiple EMAs
            e1 = df[price].ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            e3 = e2.ewm(span=length, adjust=False).mean()
            e4 = e3.ewm(span=length, adjust=False).mean()
            e5 = e4.ewm(span=length, adjust=False).mean()
            e6 = e5.ewm(span=length, adjust=False).mean()
            
            # Calculate T3
            c1 = -vfactor**3
            c2 = 3 * vfactor**2 + 3 * vfactor**3
            c3 = -6 * vfactor**2 - 3 * vfactor - 3 * vfactor**3
            c4 = 1 + 3 * vfactor + vfactor**3 + 3 * vfactor**2
            
            t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
            
            result[f'T3_{length}'] = t3
            
        # return result
        return result[f'T3_8'],result[f'T3_13']
    
    def trend_lines(self, df, length=14, price='close'):
        """
        Calculate basic trend lines (support and resistance)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for trend line calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with support and resistance lines
        """
        result = pd.DataFrame(index=df.index)
        
        # Basic Support and Resistance lines (using sliding window min/max)
        result['support'] = df[price].rolling(window=length).min()
        result['resistance'] = df[price].rolling(window=length).max()
        
        # Linear regression for trend direction
        result['trend'] = np.nan
    
        for i in range(length, len(df)):
            window = df[price].iloc[i-length:i]
            x = np.array(range(length))
            y = window.values
            
            # Calculate linear regression
            slope, intercept = np.polyfit(x, y, 1)
            result.iloc[i, result.columns.get_loc('trend')] = slope
            
        # return result
        return result['support'],result['resistance'],result['trend']
    
    def macd(self, df, fast_length=12, slow_length=26, signal_length=9, price='close'):
        """
        Moving Average Convergence Divergence
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        fast_length : int
            The window length for fast EMA calculation
        slow_length : int
            The window length for slow EMA calculation
        signal_length : int
            The window length for signal line calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with MACD, signal, and histogram values
        """
        result = pd.DataFrame(index=df.index)
        
        # Calculate fast and slow EMAs
        fast_ema = df[price].ewm(span=fast_length, adjust=False).mean()
        slow_ema = df[price].ewm(span=slow_length, adjust=False).mean()
        
        # Calculate MACD line
        # result['macd'] = fast_ema - slow_ema
        macd_ema = fast_ema - slow_ema
        
        # Calculate signal line
        # result['signal'] = result['macd'].ewm(span=signal_length, adjust=False).mean()
        signal = macd_ema.ewm(span=signal_length,adjust=False).mean()
        
        # Calculate histogram
        # result['histogram'] = result['macd'] - result['signal']
        histogram = macd_ema - signal
        
        # return result
        return macd_ema,signal,histogram
    
    def adx(self, df, length=14, adx_smoothing=14):
        """
        Average Directional Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for calculations
        adx_smoothing : int
            The smoothing period for ADX line
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with ADX, +DI, and -DI values
        """
        result = pd.DataFrame(index=df.index)
        
        # Calculate True Range
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # Calculate directional movement
        pos_dm = high - high.shift(1)
        neg_dm = low.shift(1) - low
        
        pos_dm = pos_dm.where((pos_dm > neg_dm) & (pos_dm > 0), 0)
        neg_dm = neg_dm.where((neg_dm > pos_dm) & (neg_dm > 0), 0)
        
        # Smooth with EMA
        atr = tr.ewm(span=length, adjust=False).mean()
        plus_di = 100 * (pos_dm.ewm(span=length, adjust=False).mean() / atr)
        minus_di = 100 * (neg_dm.ewm(span=length, adjust=False).mean() / atr)
        
        # Calculate Directional Index
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.ewm(span=adx_smoothing, adjust=False).mean()
        
        result['+DI'] = plus_di
        result['-DI'] = minus_di
        result['ADX'] = adx
        
        return result['+DI'],result['-DI'],result['ADX']
    
    def parabolic_sar(self, df, af_start=0.02, af_step=0.02, af_max=0.2):
        """
        Parabolic SAR
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        af_start : float
            Starting acceleration factor
        af_step : float
            Acceleration factor step
        af_max : float
            Maximum acceleration factor
            
        Returns:
        --------
        pandas.Series: Parabolic SAR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Initialize PSAR, EP, AF
        psar = pd.Series(index=df.index, dtype=float)
        ep = pd.Series(index=df.index, dtype=float)
        af = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        # Initialize first values
        psar.iloc[0] = low.iloc[0]
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = af_start
        trend.iloc[0] = 1  # Start with uptrend
        
        # Calculate PSAR
        for i in range(1, len(df)):
            # Trend transition
            if trend.iloc[i-1] == 1:  # Uptrend
                psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
                
                # Check for trend reversal
                if low.iloc[i] < psar.iloc[i]:
                    trend.iloc[i] = -1  # Switch to downtrend
                    psar.iloc[i] = ep.iloc[i-1]  # Set to prior EP
                    ep.iloc[i] = low.iloc[i]  # EP is the new low
                    af.iloc[i] = af_start  # Reset AF
                else:
                    trend.iloc[i] = 1  # Continue uptrend
                    psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_step, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                psar.iloc[i] = psar.iloc[i-1] - af.iloc[i-1] * (psar.iloc[i-1] - ep.iloc[i-1])
                
                # Check for trend reversal
                if high.iloc[i] > psar.iloc[i]:
                    trend.iloc[i] = 1  # Switch to uptrend
                    psar.iloc[i] = ep.iloc[i-1]  # Set to prior EP
                    ep.iloc[i] = high.iloc[i]  # EP is the new high
                    af.iloc[i] = af_start  # Reset AF
                else:
                    trend.iloc[i] = -1  # Continue downtrend
                    psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_step, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return psar
    
    def ichimoku(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
        """
        Ichimoku Cloud
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        tenkan_period : int
            Period for Tenkan-sen (Conversion Line)
        kijun_period : int
            Period for Kijun-sen (Base Line)
        senkou_b_period : int
            Period for Senkou Span B
        displacement : int
            Displacement for Senkou Span A and B
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with Ichimoku components
        """
        result = pd.DataFrame(index=df.index)
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        result['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        result['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_low = df['low'].rolling(window=senkou_b_period).min()
        result['senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        result['chikou_span'] = df['close'].shift(-displacement)
        
        return result
    
    def hma(self, df, length=16, price='close'):
        """
        Hull Moving Average
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for HMA calculation
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.Series: HMA values
        """
        # Calculate weighted moving averages
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        
        wma_half = self._wma(df[price], half_length)
        wma_full = self._wma(df[price], length)
        
        # Calculate raw HMA: (2 * WMA(n/2) - WMA(n))
        raw_hma = 2 * wma_half - wma_full
        
        # Calculate the HMA by taking WMA of raw_hma
        hma = self._wma(raw_hma, sqrt_length)
        
        return hma
    
    def _wma(self, series, length):
        """
        Helper function to calculate weighted moving average
        """
        weights = np.arange(1, length + 1)
        wma = series.rolling(window=length).apply(
            lambda x: np.sum(weights * x) / np.sum(weights), raw=True
        )
        return wma
    
    def donchian_channel(self, df, length=20):
        """
        Donchian Channel
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for Donchian Channel calculation
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with upper, middle, and lower bands
        """
        result = pd.DataFrame(index=df.index)
        
        result['upper'] = df['high'].rolling(window=length).max()
        result['lower'] = df['low'].rolling(window=length).min()
        result['middle'] = (result['upper'] + result['lower']) / 2
        
        return result
    
    def keltner_channel(self, df, length=20, atr_length=10, multiplier=2.0):
        """
        Keltner Channel
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        length : int
            The window length for the middle line (EMA)
        atr_length : int
            The window length for ATR calculation
        multiplier : float
            Multiplier for ATR to set channel width
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with upper, middle, and lower bands
        """
        result = pd.DataFrame(index=df.index)
        
        # Calculate middle line (EMA of typical price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle = typical_price.ewm(span=length, adjust=False).mean()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        atr = tr.ewm(span=atr_length, adjust=False).mean()
        
        # Calculate upper and lower bands
        result['middle'] = middle
        result['upper'] = middle + (multiplier * atr)
        result['lower'] = middle - (multiplier * atr)
        
        return result
    
    def gmma(self, df, short_lengths=[3, 5, 8, 10, 12, 15], long_lengths=[30, 35, 40, 45, 50, 60], price='close'):
        """
        Guppy Multiple Moving Average
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        short_lengths : list
            List of short-term EMA periods
        long_lengths : list
            List of long-term EMA periods
        price : str
            The column name in df to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with short and long EMAs
        """
        result = pd.DataFrame(index=df.index)
        
        # Calculate short-term EMAs
        for length in short_lengths:
            result[f'short_ema_{length}'] = df[price].ewm(span=length, adjust=False).mean()
        
        # Calculate long-term EMAs
        for length in long_lengths:
            result[f'long_ema_{length}'] = df[price].ewm(span=length, adjust=False).mean()
        
        return result