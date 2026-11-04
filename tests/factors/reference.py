"""
Reference implementations of factors, using pandas on the raw csv data.
They are used to check the results of spectre, independently of the engine.
"""

import numpy as np
import pandas as pd

from tests.conftest import DAILY_OHLCV, DATA_DIR

ASSETS = ("AAPL", "MSFT")
START, END = "2019-01-01", "2019-01-15"


def daily_prices(asset):
    """Raw daily prices with columns open/high/low/close/volume, aligned to AAPL's calendar."""

    def read(name):
        return pd.read_csv(DATA_DIR / "daily" / f"{name}.csv", index_col="date", parse_dates=True)

    calendar = read("AAPL").index
    df = read(asset)
    df = df[df.index.isin(calendar)][list(DAILY_OHLCV)]
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def expected(func, delayed=True, start=START, end=END):
    """
    Expected factor values of `func(prices)` of each asset, same rows as `engine.run`.
    If delayed, the values are shifted by 1 bar (of each asset), and the results start from the
    second trading day, same as the engine does.
    """
    ret = {}
    for asset in ASSETS:
        values = func(daily_prices(asset))
        values = pd.Series(values) if not isinstance(values, pd.Series) else values
        if delayed:
            values = values.shift(1)
        ret[asset] = values
    return _select(ret, delayed, start, end)


def expected_xs(func, values=lambda df: df.close, delayed=True, start=START, end=END):
    """
    Expected values of a cross-section factor, `func` receives a DataFrame of
    `values(prices)` (rows are dates, columns are assets), and returns a DataFrame of same shape.
    Like the engine, it is calculated cross-sectionally first, then delayed by each asset.
    """
    prices = {asset: daily_prices(asset) for asset in ASSETS}
    wide = pd.DataFrame({asset: values(df) for asset, df in prices.items()})
    result = func(wide)
    ret = {}
    for asset in ASSETS:
        # only the days the asset has data
        asset_values = result[asset].loc[prices[asset].index]
        ret[asset] = asset_values.shift(1) if delayed else asset_values
    return _select(ret, delayed, start, end)


def _select(series, delayed, start, end):
    calendar = series["AAPL"].loc[start:end].index
    if delayed:
        calendar = calendar[1:]
    return tuple(
        series[asset][series[asset].index.isin(calendar)].to_numpy(copy=True) for asset in ASSETS
    )


def loaded_closes(asset, start="2019-01-02", end=END):
    """Closes loaded by the engine for factors without rolling window."""
    return daily_prices(asset).close.loc[start:end].to_numpy(copy=True)


def mad_clamp(x, z, mean=False):
    """Clamp x by `center +- z * mean/median absolute deviation`"""
    x = np.asarray(x, dtype=float)
    if mean:
        center = np.nanmean(x)
        mad = np.nanmean(np.abs(x - center))
    else:
        center = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - center))
    return np.clip(x, center - z * mad, center + z * mad)


# --------------------------------- indicators ---------------------------------


def sma(x: pd.Series, win):
    return x.rolling(win).mean()


def ema(x: pd.Series, span):
    return x.ewm(span=span, adjust=False).mean()


def macd(close, fast=12, slow=26, sign=9):
    """Returns (macd, signal, histogram)"""
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sign)
    return line, signal, line - signal


def bbands(close, win=20, k=2):
    """Returns (upper, middle, lower), uses population standard deviation."""
    middle = close.rolling(win).mean()
    std = close.rolling(win).std(ddof=0)
    return middle + k * std, middle, middle - k * std


def true_range(df):
    prev_close = df.close.shift(1)
    return pd.concat(
        [df.high - df.low, (df.high - prev_close).abs(), (df.low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)


def rsi(close, win=14):
    """Cutler's RSI (simple moving average of gains and losses)."""
    diff = close.diff()
    up = diff.clip(lower=0).rolling(win).mean()
    down = (-diff.clip(upper=0)).rolling(win).mean()
    return 100 - 100 / (1 + up / down)


def stochf(df, win=14):
    """Fast stochastic oscillator %K"""
    highest = df.high.rolling(win).max()
    lowest = df.low.rolling(win).min()
    return (df.close - lowest) / (highest - lowest) * 100


def returns(close, win=2):
    return close / close.shift(win - 1) - 1


def log_returns(close, win=2):
    return np.log(close / close.shift(win - 1))
