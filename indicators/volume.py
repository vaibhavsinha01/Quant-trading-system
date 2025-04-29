import pandas as pd
import numpy as np

class VolumeIndicator:
    def __init__(self):
        pass
    
    def obv(self, df, price='close', volume='volume'):
        """
        On-Balance Volume
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        price : str
            The column name in df for price data
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: OBV values
        """
        # Calculate price direction
        price_direction = np.sign(df[price].diff())
        
        # Replace first NaN with 0
        price_direction.iloc[0] = 0
        
        # Calculate OBV: Add volume when price increases, subtract when price decreases
        obv = (price_direction * df[volume]).cumsum()
        
        return obv
    
    def vma(self, df, length=20, volume='volume'):
        """
        Volume Moving Average
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing volume data
        length : int
            The window length for moving average calculation
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: VMA values
        """
        # Calculate simple moving average of volume
        vma = df[volume].rolling(window=length).mean()
        
        return vma
    
    def vo(self, df, short_length=5, long_length=10, volume='volume'):
        """
        Volume Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing volume data
        short_length : int
            The window length for short moving average
        long_length : int
            The window length for long moving average
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with volume oscillator values and percentage
        """
        # Calculate short and long volume moving averages
        short_vma = df[volume].rolling(window=short_length).mean()
        long_vma = df[volume].rolling(window=long_length).mean()
        
        # Calculate volume oscillator
        vo_absolute = short_vma - long_vma
        vo_percent = 100 * (short_vma - long_vma) / long_vma
        
        result = pd.DataFrame({
            'vo_absolute': vo_absolute,
            'vo_percent': vo_percent
        }, index=df.index)
        
        return result
    
    def cmf(self, df, length=20):
        """
        Chaikin Money Flow
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for CMF calculation
            
        Returns:
        --------
        pandas.Series: CMF values
        """
        # Calculate Money Flow Multiplier
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate Chaikin Money Flow
        cmf = mfv.rolling(window=length).sum() / volume.rolling(window=length).sum()
        
        return cmf
    
    def adline(self, df):
        """
        Accumulation/Distribution Line
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
            
        Returns:
        --------
        pandas.Series: A/D Line values
        """
        # Calculate Money Flow Multiplier
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate A/D Line
        ad_line = mfv.cumsum()
        
        return ad_line
    
    def mfi(self, df, length=14):
        """
        Money Flow Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for MFI calculation
            
        Returns:
        --------
        pandas.Series: MFI values
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * df['volume']
        
        # Get money flow direction
        direction = np.sign(typical_price.diff())
        
        # Calculate positive and negative money flow
        positive_flow = (raw_money_flow * (direction > 0)).replace(0, np.nan)
        negative_flow = (raw_money_flow * (direction < 0)).abs().replace(0, np.nan)
        
        # Sum positive and negative money flows over the period
        positive_sum = positive_flow.rolling(window=length).sum()
        negative_sum = negative_flow.rolling(window=length).sum()
        
        # Calculate money flow ratio
        money_ratio = positive_sum / negative_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def vwap(self, df, reset_period='D'):
        """
        Volume Weighted Average Price
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        reset_period : str
            Time period to reset VWAP calculation ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
        --------
        pandas.Series: VWAP values
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price * volume
        pv = typical_price * df['volume']
        
        # Group by the reset period
        if reset_period is not None:
            # Create a grouper based on the reset period
            groups = pd.Grouper(freq=reset_period)
            
            # Group by the period and calculate cumulative sums
            cumulative_pv = pv.groupby(groups).cumsum()
            cumulative_volume = df['volume'].groupby(groups).cumsum()
            
            # Calculate VWAP
            vwap = cumulative_pv / cumulative_volume
        else:
            # No reset, calculate running VWAP
            vwap = pv.cumsum() / df['volume'].cumsum()
        
        return vwap
    
    def eom(self, df, length=14, divisor=10000):
        """
        Ease of Movement
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for EOM smoothing
        divisor : float
            Scaling factor for volume
            
        Returns:
        --------
        pandas.Series: EOM values
        """
        # Calculate distance moved
        high = df['high']
        low = df['low']
        
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        
        # Calculate box ratio
        box_ratio = (df['volume'] / divisor) / (high - low)
        
        # Replace infinite values with nan
        box_ratio = box_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate raw EOM
        raw_eom = distance / box_ratio
        
        # Smooth with moving average
        eom = raw_eom.rolling(window=length).mean()
        
        return eom
    
    def pvt(self, df, price='close', volume='volume'):
        """
        Price Volume Trend
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        price : str
            The column name in df for price data
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: PVT values
        """
        # Calculate percentage price change
        price_change_percent = df[price].pct_change()
        
        # Calculate PVT line
        pvt = (price_change_percent * df[volume]).cumsum()
        
        return pvt
    
    def fi(self, df, length=13, price='close', volume='volume'):
        """
        Force Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for EMA smoothing
        price : str
            The column name in df for price data
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: Force Index values
        """
        # Calculate raw force index
        raw_fi = df[price].diff() * df[volume]
        
        # Apply EMA smoothing
        fi = raw_fi.ewm(span=length, adjust=False).mean()
        
        return fi
    
    def kvo(self, df, short_length=34, long_length=55, signal_length=13):
        """
        Klinger Volume Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        short_length : int
            The window length for short EMA
        long_length : int
            The window length for long EMA
        signal_length : int
            The window length for signal line
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with KVO and signal values
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate trend direction
        trend = np.sign(typical_price.diff())
        
        # Calculate trend volume
        trend_volume = df['volume'] * trend
        
        # Calculate EMAs
        short_ema = trend_volume.ewm(span=short_length, adjust=False).mean()
        long_ema = trend_volume.ewm(span=long_length, adjust=False).mean()
        
        # Calculate KVO
        kvo = short_ema - long_ema
        
        # Calculate signal line
        signal = kvo.ewm(span=signal_length, adjust=False).mean()
        
        result = pd.DataFrame({
            'kvo': kvo,
            'signal': signal
        }, index=df.index)
        
        return result
    
    def vfi(self, df, length=130, coef=0.2, vcoef=2.5, smoothing_length=5):
        """
        Volume Flow Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for VFI calculation
        coef : float
            Price threshold coefficient
        vcoef : float
            Volume coefficient
        smoothing_length : int
            The window length for smoothing
            
        Returns:
        --------
        pandas.Series: VFI values
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate median price
        median_price = (df['high'] + df['low']) / 2
        
        # Calculate price cutoff (based on ATR-like calculation)
        price_range = df['high'] - df['low']
        price_cutoff = coef * price_range.rolling(window=length).mean()
        
        # Calculate price changes and directions
        price_change = median_price - median_price.shift(1)
        
        # Apply price cutoff to determine significant moves
        price_direction = np.where(
            price_change > price_cutoff, 1,
            np.where(price_change < -price_cutoff, -1, 0)
        )
        
        # Calculate inter-day volatility
        inter_day_vol = np.log(df['volume'] / df['volume'].shift(1))
        inter_day_vol.fillna(0, inplace=True)
        
        # Calculate volume adjustment factor
        volume_adj = inter_day_vol.rolling(window=length).std() * vcoef
        
        # Apply volume adjustments
        adjusted_volume = np.where(
            inter_day_vol > volume_adj, 
            np.log(df['volume']), 
            np.where(
                inter_day_vol < -volume_adj, 
                -np.log(df['volume']), 
                0
            )
        )
        
        # Calculate raw VFI
        raw_vfi = np.cumsum(price_direction * adjusted_volume)
        
        # Apply smoothing
        vfi = pd.Series(raw_vfi, index=df.index).ewm(span=smoothing_length, adjust=False).mean()
        
        return vfi
    
    def iii(self, df):
        """
        Intraday Intensity Index
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
            
        Returns:
        --------
        pandas.Series: III values
        """
        # Calculate III
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        numerator = 2 * close - high - low
        denominator = (high - low) * volume
        
        iii = numerator / denominator
        iii = iii.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
        
        return iii
    
    def nv(self, df, length=14, volume='volume'):
        """
        Normalized Volume
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing volume data
        length : int
            The window length for normalization
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: Normalized Volume values
        """
        # Calculate average volume over the period
        avg_volume = df[volume].rolling(window=length).mean()
        
        # Calculate normalized volume
        normalized_volume = df[volume] / avg_volume
        
        return normalized_volume
    
    def rvol(self, df, length=20, volume='volume'):
        """
        Relative Volume
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing volume data
        length : int
            The window length for average volume calculation
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: Relative Volume values
        """
        # Calculate average volume
        avg_volume = df[volume].rolling(window=length).mean()
        
        # Calculate relative volume
        relative_volume = df[volume] / avg_volume
        
        return relative_volume
    
    def dv(self, df, length=1, volume='volume'):
        """
        Delta Volume
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing volume data
        length : int
            The difference period
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: Delta Volume values
        """
        # Calculate volume change
        delta_volume = df[volume] - df[volume].shift(length)
        
        return delta_volume
    
    def tsv(self, df, length=13, price='close', volume='volume'):
        """
        Time Segmented Volume
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for TSV calculation
        price : str
            The column name in df for price data
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: TSV values
        """
        # Calculate price change
        price_change = df[price] - df[price].shift(1)
        
        # Calculate volume * price change
        vol_price_change = price_change * df[volume]
        
        # Calculate TSV
        tsv = vol_price_change.rolling(window=length).sum()
        
        return tsv
    
    def vzo(self, df, length=14, ratio_length=6, volume='volume'):
        """
        Volume Zone Oscillator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        length : int
            The window length for VZO calculation
        ratio_length : int
            The window length for EMA of volume ratio
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: VZO values
        """
        # Calculate price direction
        close_change = df['close'].diff()
        
        # Calculate positive and negative volume
        positive_volume = np.where(close_change > 0, df[volume], 0)
        negative_volume = np.where(close_change < 0, df[volume], 0)
        
        # Convert to series
        positive_volume = pd.Series(positive_volume, index=df.index)
        negative_volume = pd.Series(negative_volume, index=df.index)
        
        # Calculate EMAs of positive and negative volume
        ema_pos_vol = positive_volume.ewm(span=length, adjust=False).mean()
        ema_neg_vol = negative_volume.ewm(span=length, adjust=False).mean()
        
        # Calculate volume ratio
        volume_ratio = (ema_pos_vol - ema_neg_vol) / (ema_pos_vol + ema_neg_vol)
        
        # Apply smoothing (EMA)
        vzo = 100 * volume_ratio.ewm(span=ratio_length, adjust=False).mean()
        
        return vzo
    
    def tvi(self, df, min_tick=0.01, length=14, volume='volume'):
        """
        Tick Volume Indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        min_tick : float
            Minimum price movement
        length : int
            The window length for TVI calculation
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.Series: TVI values
        """
        # Calculate price movement
        price_movement = abs(df['close'] - df['close'].shift(1))
        
        # Calculate tick volume (estimated number of trades)
        tick_volume = price_movement / min_tick
        
        # Calculate ratio of actual volume to tick volume
        volume_per_tick = df[volume] / tick_volume.replace(0, np.nan)
        
        # Apply smoothing
        tvi = volume_per_tick.rolling(window=length).mean()
        
        return tvi
    
    def cdv(self, df, price='close', volume='volume'):
        """
        Cumulative Delta Volume
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        price : str
            The column name in df for price data
        volume : str
            The column name in df for volume data
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with delta volume and cumulative delta volume
        """
        # Calculate price change direction
        price_direction = np.sign(df[price].diff())
        
        # Calculate delta volume (positive when price increases, negative when decreases)
        delta_volume = price_direction * df[volume]
        
        # Calculate cumulative delta volume
        cum_delta_volume = delta_volume.cumsum()
        
        result = pd.DataFrame({
            'delta_volume': delta_volume,
            'cumulative_delta_volume': cum_delta_volume
        }, index=df.index)
        
        return result