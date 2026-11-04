"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

from .arrow import ArrowLoader
from .csv import CsvDirLoader
from .sp500 import sp500_constituents


def _to_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo's prices, volumes and dividends are already adjusted for all later splits. Undo that and
    add the columns `DataLoader` adjusts with, because giving it the splits of adjusted prices
    would adjust them twice.
    """
    # 'Stock Splits' is to/from, like 4 for a 4-for-1 split, and 0 on the days without a split
    splits = df["Stock Splits"].fillna(0).replace(0, 1)
    later_splits = splits[::-1].cumprod()[::-1].shift(-1, fill_value=1)
    raw = df[["Open", "High", "Low", "Close"]].mul(later_splits, axis=0)
    raw["Volume"] = df["Volume"] / later_splits
    # A dividend is paid on the shares before its ex-date, so include a split on that same date
    raw["ex-dividend"] = df["Dividends"].fillna(0) * later_splits * splits
    raw["split_ratio"] = 1 / splits
    return raw


class YahooDownloader:
    # Symbols of one `yfinance.download`, `skip_exists` resumes from the last finished batch.
    batch_size = 100
    # Seconds to wait before downloading the failed symbols again.
    retry_delays = (5, 10)

    @classmethod
    def ingest(
        cls, start_date: str, save_to: str, symbols: list | None = None, skip_exists=True
    ) -> None:
        """
        Download daily data from yahoo. Prices are raw, as they were traded on that day, and come
        with dividends and splits, so factors and backtests can adjust them.

        Yahoo has no data for delisted stocks, so any universe made of today's stocks has
        survivorship bias. `sp500_constituents` has the members of the past, but only about half
        of the members of 2001 can still be downloaded.

        :param start_date:
        :param save_to: path to folder
        :param symbols: list of symbol to download. If is None, download the latest S&P 500
            components, plus SPY and QQQ.
        :param skip_exists: skip if file exists, useful for resume from interruption.
        """
        import yfinance
        from tqdm.auto import tqdm

        calender_asset = None
        if symbols is None:
            warnings.warn(
                "`symbols=None` downloads the latest S&P 500 components, and yahoo has no data for "
                "delisted stocks, so backtests on this data have survivorship bias.",
                RuntimeWarning,
            )
            symbols = [*sorted(sp500_constituents().iloc[-1]), "SPY", "QQQ"]
            calender_asset = "SPY"

        prices_dir = os.path.join(save_to, "daily")
        os.makedirs(prices_dir, exist_ok=True)

        def csv_path(symbol):
            return os.path.join(prices_dir, f"{symbol}.csv")

        failed = [symbol.replace(".", "-").upper() for symbol in symbols]
        if skip_exists:
            failed = [symbol for symbol in failed if not os.path.exists(csv_path(symbol))]

        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        # `end` is exclusive, so this leaves out today's bar, which is unfinished during trading
        end = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")

        def download(batch):
            """Save the symbols in `batch`, and return the ones without data."""
            # It doesn't raise, the symbols that failed for any reason just have no prices.
            data = yfinance.download(
                batch,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
            )
            missing = []
            for symbol in batch:
                prices = data[symbol].dropna(subset=["Close"])
                if prices.empty:
                    missing.append(symbol)
                else:
                    _to_raw_prices(prices).to_csv(csv_path(symbol))
            return missing

        print("Download prices from yahoo...")
        for delay in (*cls.retry_delays, None):
            size = cls.batch_size
            batches = [failed[i : i + size] for i in range(0, len(failed), size)]
            failed = [symbol for batch in tqdm(batches) for symbol in download(batch)]
            if not failed or delay is None:
                break
            time.sleep(delay)
        if failed:
            print(f"No data from yahoo, skipped: {failed}")

        print("Converting...")
        loader = CsvDirLoader(
            prices_dir,
            calender_asset=calender_asset,
            ohlcv=("Open", "High", "Low", "Close", "Volume"),
            adjustments=("ex-dividend", "split_ratio"),
            prices_index="Date",
            parse_dates=True,
            dtype={
                "Open": np.float32,
                "High": np.float32,
                "Low": np.float32,
                "Close": np.float32,
                "Volume": np.float64,
                "ex-dividend": np.float64,
                "split_ratio": np.float64,
            },
        )

        arrow_file = os.path.join(save_to, "yahoo.feather")
        ArrowLoader.ingest(source=loader, save_to=arrow_file, force=True)

        print(
            f'Ingest completed! Use `loader = spectre.data.ArrowLoader(r"{arrow_file}")` '
            "to load your data."
        )
