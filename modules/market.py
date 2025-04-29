# this file tells how the market is going now

import numpy as np
import pandas as pd

def hurst_exponent(df, column='close', max_lag=100):
    """
    Calculates the Hurst Exponent of a time series from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with price data.
    - column (str): The name of the column to analyze. Default is 'close'.
    - max_lag (int): The maximum lag to use for calculation.

    Returns:
    - float: Estimated Hurst Exponent
    """
    ts = df[column].values
    lags = range(2, max_lag)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst
