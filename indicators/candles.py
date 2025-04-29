import candlestick.candlestick
import pandas as pd
import numpy as np
from candlestick import candlestick

class CandleIndicators:
    def __init__(self):
        pass
    
    def marubozu(self, df):
        df['cdl_marubozu'] = np.where(
            (np.abs(df['open'] - df['low']) < 0.001) & (np.abs(df['close'] - df['high']) < 0.001), 1,
            np.where((np.abs(df['open'] - df['high']) < 0.001) & (np.abs(df['close'] - df['low']) < 0.001), -1, 0)
        )
        return df

    def doji(self, df):
        return candlestick.doji(df)

    def inverted_hammer(self, df):
        return candlestick.inverted_hammer(df)

    def hammer(self, df):
        return candlestick.hammer(df)

    def shooting_star(self, df):
        return candlestick.shooting_star(df)

    def hanging_man(self, df):
        return candlestick.hanging_man(df)

    def bullish_engulfing(self, df):
        return candlestick.bullish_engulfing(df)

    def bearish_engulfing(self, df):
        return candlestick.bearish_engulfing(df)

    def morning_star(self, df):
        return candlestick.morning_star(df)

    def evening_star(self, df):
        df['cdl_evening_star'] = 0
        for i in range(2, len(df)):
            p1 = df.iloc[i-2]
            p2 = df.iloc[i-1]
            p3 = df.iloc[i]
            if (p1['close'] > p1['open'] and
                abs(p2['close'] - p2['open']) < (p1['close'] - p1['open']) * 0.3 and
                p3['close'] < p3['open'] and p3['close'] < p1['open']):
                df.at[i, 'cdl_evening_star'] = -1

        return df

    def piercing_pattern(self, df):
        return candlestick.piercing_pattern(df)

    def dark_cloud_cover(self, df):
        return candlestick.dark_cloud_cover(df)

    def spinning_top(self, df, body_threshold=0.3, shadow_ratio=2.0):
        """
        Identifies spinning top candles in the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Must contain columns 'Open', 'High', 'Low', 'Close'.
            body_threshold (float): Max ratio of real body to full candle range.
            shadow_ratio (float): Min ratio of each shadow to real body.
        
        Returns:
            pd.Series: Boolean Series where True indicates a spinning top candle.
        """
        open_ = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        
        body = (close - open_).abs()
        upper_shadow = high - open_.where(close > open_, close)
        lower_shadow = open_.where(close > open_, close) - low
        full_range = high - low

        # Avoid division by zero
        body = body.replace(0, 1e-6)

        condition_body_small = body / full_range < body_threshold
        condition_upper_long = upper_shadow / body > shadow_ratio
        condition_lower_long = lower_shadow / body > shadow_ratio

        return condition_body_small & condition_upper_long & condition_lower_long


    def three_white_soldiers(self, df):
        df['cdl_three_white_soldiers'] = 0
        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1]
            c3 = df.iloc[i]
            if (c1['close'] > c1['open'] and
                c2['close'] > c2['open'] and c2['open'] > c1['open'] and c2['close'] > c1['close'] and
                c3['close'] > c3['open'] and c3['open'] > c2['open'] and c3['close'] > c2['close']):
                df.at[i, 'cdl_three_white_soldiers'] = 1

    def three_black_crows(self, df):
        df['cdl_three_black_crows'] = 0
        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1]
            c3 = df.iloc[i]
            if (c1['close'] < c1['open'] and
                c2['close'] < c2['open'] and c2['open'] < c1['open'] and c2['close'] < c1['close'] and
                c3['close'] < c3['open'] and c3['open'] < c2['open'] and c3['close'] < c2['close']):
                df.at[i, 'cdl_three_black_crows'] = -1
    
    def evening_star(self, df):
        df['cdl_evening_star'] = 0
        for i in range(2, len(df)):
            p1 = df.iloc[i-2]
            p2 = df.iloc[i-1]
            p3 = df.iloc[i]
            if (p1['close'] > p1['open'] and
                abs(p2['close'] - p2['open']) < (p1['close'] - p1['open']) * 0.3 and
                p3['close'] < p3['open'] and p3['close'] < p1['open']):
                df.at[i, 'cdl_evening_star'] = -1 

    def rickshaw_man(self, df):                                                                                                                        
        body = np.abs(df['close'] - df['open'])
        upper = df['high'] - np.maximum(df['open'], df['close'])
        lower = np.minimum(df['open'], df['close']) - df['low']
        total_range = df['high'] - df['low']
        df['cdl_rickshaw_man'] = np.where((body / total_range < 0.1) & (upper > total_range * 0.4) & (lower > total_range * 0.4), 1, 0)

    def high_wave(self, df):
        body = np.abs(df['close'] - df['open'])
        upper = df['high'] - np.maximum(df['open'], df['close'])
        lower = np.minimum(df['open'], df['close']) - df['low']
        df['cdl_high_wave'] = np.where((body < upper) & (body < lower) & (upper > 0.03) & (lower > 0.03), 1, 0)

    def matching_low(self, df):
        df['cdl_matching_low'] = 0
        for i in range(1, len(df)):
            if df['close'][i] == df['close'][i-1] and df['close'][i] < df['open'][i] and df['close'][i-1] < df['open'][i-1]:
                df.at[i, 'cdl_matching_low'] = 1

    def matching_high(self, df):
        df['cdl_matching_high'] = 0
        for i in range(1, len(df)):
            if df['close'][i] == df['close'][i-1] and df['close'][i] > df['open'][i] and df['close'][i-1] > df['open'][i-1]:
                df.at[i, 'cdl_matching_high'] = -1

    def tasuki_gap(self, df):
        df['cdl_tasuki_gap'] = 0
        for i in range(2, len(df)):
            prev_gap = df['open'][i-2] > df['close'][i-2] and df['open'][i-1] > df['close'][i-1]
            if prev_gap and df['open'][i] < df['close'][i]:
                df.at[i, 'cdl_tasuki_gap'] = 1

    def advance_block(self, df):
        df['cdl_advance_block'] = 0
        for i in range(2, len(df)):
            if (df['close'][i-2] > df['open'][i-2] and
                df['close'][i-1] > df['open'][i-1] and df['close'][i-1] > df['close'][i-2] and
                df['close'][i] > df['open'][i] and df['close'][i] > df['close'][i-1] and
                df['high'][i] - df['close'][i] > df['close'][i] - df['open'][i]):
                df.at[i, 'cdl_advance_block'] = 1

    def deliberation(self, df):
        df['cdl_deliberation'] = 0
        for i in range(2, len(df)):
            if (df['close'][i] > df['close'][i-1] > df['close'][i-2] and
                df['close'][i] - df['open'][i] < df['close'][i-1] - df['open'][i-1]):
                df.at[i, 'cdl_deliberation'] = 1

    def tristar(self, df):
        df['cdl_tristar'] = 0
        for i in range(2, len(df)):
            if (np.abs(df['close'][i-2] - df['open'][i-2]) < 0.1 and
                np.abs(df['close'][i-1] - df['open'][i-1]) < 0.1 and
                np.abs(df['close'][i] - df['open'][i]) < 0.1):
                df.at[i, 'cdl_tristar'] = 1

    def kicker(self, df):
        df['cdl_kicker'] = 0
        for i in range(1, len(df)):
            if (df['close'][i-1] < df['open'][i-1] and df['open'][i] > df['close'][i-1] and df['close'][i] > df['open'][i]):
                df.at[i, 'cdl_kicker'] = 1

    def separating_lines(self, df):
        df['cdl_separating_lines'] = 0
        for i in range(1, len(df)):
            if (df['close'][i-1] > df['open'][i-1] and df['open'][i] == df['open'][i-1] and df['close'][i] > df['open'][i]):
                df.at[i, 'cdl_separating_lines'] = 1

    def stick_sandwich(self, df):
        df['cdl_stick_sandwich'] = 0
        for i in range(2, len(df)):
            if (df['close'][i-2] < df['open'][i-2] and
                df['close'][i-1] > df['open'][i-1] and
                df['close'][i] < df['open'][i] and
                df['close'][i] == df['close'][i-2]):
                df.at[i, 'cdl_stick_sandwich'] = 1

    def three_inside_up(self, df):
        df['cdl_three_inside_up'] = 0
        for i in range(2, len(df)):
            if (df['close'][i-2] < df['open'][i-2] and
                df['close'][i-1] > df['open'][i-1] and df['close'][i-1] > df['open'][i-2] and
                df['close'][i] > df['close'][i-1]):
                df.at[i, 'cdl_three_inside_up'] = 1

    def three_inside_down(self, df):
        df['cdl_three_inside_down'] = 0
        for i in range(2, len(df)):
            if (df['close'][i-2] > df['open'][i-2] and
                df['close'][i-1] < df['open'][i-1] and df['close'][i-1] < df['open'][i-2] and
                df['close'][i] < df['close'][i-1]):
                df.at[i, 'cdl_three_inside_down'] = -1

    def tower_top(self, df):
        df['cdl_tower_top'] = 0
        for i in range(2, len(df)):
            if (df['close'][i-2] > df['open'][i-2] and
                np.abs(df['close'][i-1] - df['open'][i-1]) < 0.1 and
                df['close'][i] < df['open'][i]):
                df.at[i, 'cdl_tower_top'] = -1

    def ladder_bottom(self, df):
        df['cdl_ladder_bottom'] = 0
        for i in range(3, len(df)):
            if (df['close'][i-3] < df['open'][i-3] and
                df['close'][i-2] < df['open'][i-2] and
                df['close'][i-1] > df['open'][i-1] and
                df['close'][i] > df['close'][i-1]):
                df.at[i, 'cdl_ladder_bottom'] = 1

    def upside_gap_two_crows(self, df):
        df['cdl_upside_gap_two_crows'] = 0
        for i in range(2, len(df)):
            if (df['close'][i-2] > df['open'][i-2] and
                df['open'][i-1] > df['high'][i-2] and df['close'][i-1] < df['open'][i-1] and
                df['open'][i] > df['close'][i-1] and df['close'][i] < df['open'][i]):
                df.at[i, 'cdl_upside_gap_two_crows'] = -1


if __name__ == "__main__":
    # Sample DataFrame with dummy candlestick data
    data = {
        'open': [100, 102, 101, 105, 106, 108, 107, 110],
        'high': [105, 104, 103, 108, 109, 111, 110, 113],
        'low': [98, 100, 99, 102, 103, 106, 105, 108],
        'close': [102, 101, 103, 107, 104, 109, 106, 111]
    }
    df = pd.DataFrame(data)

    # Instantiate the indicator class
    ci = CandleIndicators()

    # Apply methods that return updated DataFrames
    df = ci.marubozu(df)
    df['cdl_spinning_top'] = ci.spinning_top(df)

    # Apply all in-place modification methods
    ci.rickshaw_man(df)
    ci.high_wave(df)
    ci.matching_low(df)
    ci.matching_high(df)
    ci.tasuki_gap(df)
    ci.advance_block(df)
    ci.deliberation(df)
    ci.tristar(df)
    ci.kicker(df)
    ci.separating_lines(df)
    ci.stick_sandwich(df)
    ci.three_inside_up(df)
    ci.three_inside_down(df)
    ci.tower_top(df)
    ci.ladder_bottom(df)
    ci.upside_gap_two_crows(df)
    ci.three_white_soldiers(df)
    ci.three_black_crows(df)
    ci.evening_star(df)

    # Display the resulting DataFrame with pattern columns
    pd.set_option('display.max_columns', None)
    print(df)

