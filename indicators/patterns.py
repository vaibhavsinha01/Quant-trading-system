import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

class Patterns:
    def __init__(self, df):
        self.df = df.copy()
        self.df.reset_index(drop=True, inplace=True)

    def _get_local_extrema(self, column='close', order=5):
        highs = argrelextrema(self.df[column].values, np.greater, order=order)[0]
        lows = argrelextrema(self.df[column].values, np.less, order=order)[0]
        return highs, lows

    def m_top(self, distance=5, prominence=0.01):
        highs, _ = self._get_local_extrema(order=distance)
        m_tops = []

        for i in range(1, len(highs) - 1):
            l, m, r = highs[i-1], highs[i], highs[i+1]
            if abs(self.df.loc[l, 'close'] - self.df.loc[r, 'close']) / self.df.loc[m, 'close'] < prominence:
                if self.df.loc[m, 'close'] > self.df.loc[l, 'close'] and self.df.loc[m, 'close'] > self.df.loc[r, 'close']:
                    m_tops.append(m)

        self.df['m_top'] = 0
        self.df.loc[m_tops, 'm_top'] = 1
        return self.df

    def w_bottom(self, distance=5, prominence=0.01):
        _, lows = self._get_local_extrema(order=distance)
        w_bottoms = []

        for i in range(1, len(lows) - 1):
            l, m, r = lows[i-1], lows[i], lows[i+1]
            if abs(self.df.loc[l, 'close'] - self.df.loc[r, 'close']) / self.df.loc[m, 'close'] < prominence:
                if self.df.loc[m, 'close'] < self.df.loc[l, 'close'] and self.df.loc[m, 'close'] < self.df.loc[r, 'close']:
                    w_bottoms.append(m)

        self.df['w_bottom'] = 0
        self.df.loc[w_bottoms, 'w_bottom'] = 1
        return self.df

    def head_and_shoulders(self, distance=5, threshold=0.01):
        highs, _ = self._get_local_extrema(order=distance)
        hs_indices = []

        for i in range(2, len(highs) - 2):
            ls, h, rs = highs[i-2], highs[i], highs[i+2]
            if (self.df.loc[h, 'close'] > self.df.loc[ls, 'close'] and
                self.df.loc[h, 'close'] > self.df.loc[rs, 'close'] and
                abs(self.df.loc[ls, 'close'] - self.df.loc[rs, 'close']) / self.df.loc[h, 'close'] < threshold):
                hs_indices.append(h)

        self.df['head_and_shoulders'] = 0
        self.df.loc[hs_indices, 'head_and_shoulders'] = 1
        return self.df

    def inverse_head_and_shoulders(self, distance=5, threshold=0.01):
        _, lows = self._get_local_extrema(order=distance)
        ihs_indices = []

        for i in range(2, len(lows) - 2):
            ls, h, rs = lows[i-2], lows[i], lows[i+2]
            if (self.df.loc[h, 'close'] < self.df.loc[ls, 'close'] and
                self.df.loc[h, 'close'] < self.df.loc[rs, 'close'] and
                abs(self.df.loc[ls, 'close'] - self.df.loc[rs, 'close']) / self.df.loc[h, 'close'] < threshold):
                ihs_indices.append(h)

        self.df['inverse_head_and_shoulders'] = 0
        self.df.loc[ihs_indices, 'inverse_head_and_shoulders'] = 1
        return self.df

    def triple_top(self, distance=5, tolerance=0.01):
        highs, _ = self._get_local_extrema(order=distance)
        ttops = []

        for i in range(2, len(highs)):
            p1, p2, p3 = highs[i-2:i+1]
            prices = [self.df.loc[p, 'close'] for p in [p1, p2, p3]]
            if max(prices) - min(prices) < tolerance * prices[1]:
                ttops.append(p2)

        self.df['triple_top'] = 0
        self.df.loc[ttops, 'triple_top'] = 1
        return self.df

    def triple_bottom(self, distance=5, tolerance=0.01):
        _, lows = self._get_local_extrema(order=distance)
        tbottoms = []

        for i in range(2, len(lows)):
            p1, p2, p3 = lows[i-2:i+1]
            prices = [self.df.loc[p, 'close'] for p in [p1, p2, p3]]
            if max(prices) - min(prices) < tolerance * prices[1]:
                tbottoms.append(p2)

        self.df['triple_bottom'] = 0
        self.df.loc[tbottoms, 'triple_bottom'] = 1
        return self.df

    def cup_handle(self, distance=10, depth_ratio=0.1):
        _, lows = self._get_local_extrema(order=distance)
        ch_indices = []

        for i in range(2, len(lows)):
            l, m, r = lows[i-2], lows[i-1], lows[i]
            left, mid, right = self.df.loc[l, 'close'], self.df.loc[m, 'close'], self.df.loc[r, 'close']
            if left > mid < right and abs(left - right) / mid < depth_ratio:
                ch_indices.append(m)

        self.df['cup_handle'] = 0
        self.df.loc[ch_indices, 'cup_handle'] = 1
        return self.df

    def rounding_bottom(self, window=10, curvature_threshold=0.5):
        rb_indices = []
        for i in range(window, len(self.df) - window):
            window_data = self.df['close'].iloc[i - window:i + window + 1].values
            x = np.arange(len(window_data))
            fit = np.polyfit(x, window_data, 2)
            if fit[0] > curvature_threshold:  # upward parabola
                rb_indices.append(i)

        self.df['rounding_bottom'] = 0
        self.df.loc[rb_indices, 'rounding_bottom'] = 1
        return self.df

    def rectangle_pattern(self, window=10, tolerance=0.01):
        rect_indices = []
        for i in range(window, len(self.df)):
            sub = self.df['close'].iloc[i - window:i]
            if sub.max() - sub.min() < tolerance * sub.mean():
                rect_indices.append(i)

        self.df['rectangle_pattern'] = 0
        self.df.loc[rect_indices, 'rectangle_pattern'] = 1
        return self.df

    def ascending_triangle(self, window=10, tolerance=0.01):
        at_indices = []
        for i in range(window, len(self.df)):
            sub = self.df['close'].iloc[i - window:i]
            upper = sub.max()
            trend = np.polyfit(range(len(sub)), sub, 1)
            if trend[0] > 0 and abs(upper - sub.iloc[-1]) / upper < tolerance:
                at_indices.append(i)

        self.df['ascending_triangle'] = 0
        self.df.loc[at_indices, 'ascending_triangle'] = 1
        return self.df

    def descending_triangle(self, window=10, tolerance=0.01):
        dt_indices = []
        for i in range(window, len(self.df)):
            sub = self.df['close'].iloc[i - window:i]
            lower = sub.min()
            trend = np.polyfit(range(len(sub)), sub, 1)
            if trend[0] < 0 and abs(lower - sub.iloc[-1]) / lower < tolerance:
                dt_indices.append(i)

        self.df['descending_triangle'] = 0
        self.df.loc[dt_indices, 'descending_triangle'] = 1
        return self.df

    def symmetric_triangle(self, window=10, slope_diff=0.05):
        st_indices = []
        for i in range(window, len(self.df)):
            sub = self.df['close'].iloc[i - window:i].reset_index(drop=True)
            
            # Smooth the highs and lows using a rolling window
            highs = sub.rolling(3).max().dropna().reset_index(drop=True)
            lows = sub.rolling(3).min().dropna().reset_index(drop=True)

            if len(highs) != len(lows):  # Just in case
                continue

            x = np.arange(len(highs))
            if len(x) < 2:  # Avoid trying to fit too few points
                continue

            upper_trend = np.polyfit(x, highs, 1)[0]
            lower_trend = np.polyfit(x, lows, 1)[0]

            if upper_trend < 0 and lower_trend > 0 and abs(upper_trend - lower_trend) < slope_diff:
                st_indices.append(i)

        self.df['symmetric_triangle'] = 0
        self.df.loc[st_indices, 'symmetric_triangle'] = 1
        return self.df


if __name__ == "__main__":

    # Create sample data
    dates = pd.date_range(start="2022-01-01", periods=200)
    np.random.seed(1)
    prices = np.cumsum(np.random.randn(200)) + 100  # Simulated price series

    df = pd.DataFrame({
        "date": dates,
        "close": prices
    })

    # Instantiate Patterns class
    pattern_detector = Patterns(df)

    # Apply all pattern methods
    df = pattern_detector.m_top()
    df = pattern_detector.w_bottom()
    df = pattern_detector.head_and_shoulders()
    df = pattern_detector.inverse_head_and_shoulders()
    df = pattern_detector.triple_top()
    df = pattern_detector.triple_bottom()
    df = pattern_detector.cup_handle()
    df = pattern_detector.rounding_bottom()
    df = pattern_detector.rectangle_pattern()
    df = pattern_detector.ascending_triangle()
    df = pattern_detector.descending_triangle()
    df = pattern_detector.symmetric_triangle()

    patterns = [
        ('m_top', 'M Top', 'red', '^'),
        ('w_bottom', 'W Bottom', 'green', 'v'),
        ('head_and_shoulders', 'Head & Shoulders', 'blue', '^'),
        ('inverse_head_and_shoulders', 'Inverse H&S', 'orange', 'v'),
        ('triple_top', 'Triple Top', 'purple', '^'),
        ('triple_bottom', 'Triple Bottom', 'brown', 'v'),
        ('cup_handle', 'Cup & Handle', 'pink', 'o'),
        ('rounding_bottom', 'Rounding Bottom', 'cyan', 'o'),
        ('rectangle_pattern', 'Rectangle', 'gray', 's'),
        ('ascending_triangle', 'Ascending Triangle', 'olive', '^'),
        ('descending_triangle', 'Descending Triangle', 'teal', 'v'),
        ('symmetric_triangle', 'Symmetric Triangle', 'black', 'x'),
    ]

    plt.figure(figsize=(24, 20))
    for idx, (col, title, color, marker) in enumerate(patterns, 1):
        plt.subplot(4, 3, idx)
        plt.plot(df['date'], df['close'], label='Close Price', linewidth=1)
        plt.scatter(df.loc[df[col] == 1, 'date'], df.loc[df[col] == 1, 'close'],
                    color=color, label=title, marker=marker)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

