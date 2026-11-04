"""
Shared fixtures for the spectre test-suite.

Run all tests in parallel:  ``pytest -n auto``  (needs pytest-xdist)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import spectre

DATA_DIR = Path(__file__).parent / "data"

# The test data is a slice of IEX data of AAPL and MSFT, with some deliberate defects:
#   * MSFT has an all-NaN row on 2019-01-05 (Saturday), it is dropped when the loader uses
#     `calender_asset='AAPL'`.
#   * MSFT has no data on 2019-01-10.
DAILY_OHLCV = ("uOpen", "uHigh", "uLow", "uClose", "uVolume")
INTRADAY_OHLCV = ("Open", "High", "Low", "Close", "Volume")

# Datasets which are too big to be included in this repository, tests using them are skipped
# when they are not found. Defaults to `../historical_data/us/prices/quandl` next to the repo.
EXTERNAL_DATA_DIR = Path(
    os.environ.get(
        "SPECTRE_EXTERNAL_DATA",
        Path(__file__).parents[2] / "historical_data" / "us" / "prices" / "quandl",
    )
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that download data from the internet",
    )
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    skip_cuda = pytest.mark.skip(reason="CUDA is not available")
    skip_network = pytest.mark.skip(reason="needs internet, use --run-network to run")
    skip_slow = pytest.mark.skip(
        reason="TODO: slow test, split and simplify it. Use --run-slow to run"
    )
    has_cuda = torch.cuda.is_available()
    run_network = config.getoption("--run-network")
    run_slow = config.getoption("--run-slow")
    for item in items:
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


# ------------------------------------ devices ------------------------------------


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    """Run the test on cpu, and on cuda if available."""
    return request.param


# ------------------------------------ loaders ------------------------------------
# Loaders don't cache anything, so they can be safely shared within a test session.


def make_daily_loader(**kwargs):
    params = dict(ohlcv=DAILY_OHLCV, prices_index="date", parse_dates=True)
    params.update(kwargs)
    return spectre.data.CsvDirLoader(str(DATA_DIR / "daily"), **params)


@pytest.fixture(scope="session")
def daily_loader():
    """Daily prices of AAPL and MSFT, aligned to the trading calendar of AAPL."""
    return make_daily_loader(calender_asset="AAPL")


@pytest.fixture(scope="session")
def daily_loader_no_calendar():
    """Daily prices, without calendar alignment, so the junk row of MSFT (2019-01-05) is kept."""
    return make_daily_loader()


def make_adjusted_daily_loader(**kwargs):
    return make_daily_loader(
        dividends_path=str(DATA_DIR / "dividends"),
        splits_path=str(DATA_DIR / "splits"),
        adjustments=("amount", "ratio"),
        dividends_index="exDate",
        splits_index="exDate",
        **kwargs,
    )


@pytest.fixture(scope="session")
def adjusted_daily_loader():
    """Daily prices with dividends and splits (MSFT 1-for-15 reverse split on 2019-01-14)."""
    return make_adjusted_daily_loader(calender_asset="AAPL")


@pytest.fixture(scope="session")
def adjusted_daily_loader_no_calendar():
    return make_adjusted_daily_loader()


@pytest.fixture(scope="session")
def intraday_loader():
    """5 minutes bars of AAPL and MSFT, from 2018-12 to 2019-01."""
    return spectre.data.CsvDirLoader(
        str(DATA_DIR / "5mins"),
        prices_by_year=True,
        prices_index="Date",
        ohlcv=INTRADAY_OHLCV,
        parse_dates=True,
    )


# ------------------------------------ engines ------------------------------------


@pytest.fixture
def engine(daily_loader):
    return spectre.factors.FactorEngine(daily_loader)


@pytest.fixture
def engine_no_calendar(daily_loader_no_calendar):
    return spectre.factors.FactorEngine(daily_loader_no_calendar)


# ------------------------------------ dirty data -----------------------------------
# The csv data above is nearly clean, so it can't show how a factor treats missing data. This
# panel has one asset per defect, so a test can choose the defect it is about.

DIRTY_OHLCV = ("open", "high", "low", "close", "volume")
DIRTY_ADJUSTMENTS = ("ex-dividend", "split_ratio")
DIRTY_DATES = pd.date_range("2020-01-01", periods=60, freq="B", tz="UTC")
DIRTY_SECTORS = {
    "FULL": 1,
    "LISTED": 1,
    "DELISTED": 2,
    "HALTED": 2,
    "NANBAR": 3,
    "SPLIT": 3,
    "NANSPLIT": 3,
}
# Positions in `DIRTY_DATES`.
LISTED_ON = 30
DELISTED_ON = 40
HALTED_BARS = slice(20, 32)
NAN_BAR = 25
SPLIT_ON = 35
DIVIDEND_ON = 45
NANSPLIT_ON = 36


def make_dirty_panel():
    """
    Daily bars of these assets, indexed by ['date', 'asset']:

    ========  ======================================================================
    FULL      no defects
    LISTED    no rows before bar `LISTED_ON`
    DELISTED  no rows after bar `DELISTED_ON`
    HALTED    no rows on `HALTED_BARS`
    NANBAR    NaN prices on bar `NAN_BAR`, and a valid volume
    SPLIT     2-for-1 split on bar `SPLIT_ON`, dividend of 0.5 on bar `DIVIDEND_ON`
    NANSPLIT  2-for-1 split on bar `NANSPLIT_ON`, and NaN prices on the bar before it
    ========  ======================================================================

    Prices are raw. `clean_close` is the close if the split and the dividend never happened,
    so `close / clean_close` is the scale of each bar, and an adjusted window moves like
    `clean_close` in the scale of its last bar. `sector` is an integer column for `groupby`.
    """
    rng = np.random.default_rng(0)
    bars = len(DIRTY_DATES)
    frames = {}
    for asset, sector in DIRTY_SECTORS.items():
        clean = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, bars))
        split_scale, dividend_scale = np.ones(bars), np.ones(bars)
        dividend, split_ratio = np.zeros(bars), np.ones(bars)
        if asset == "SPLIT":
            split_ratio[SPLIT_ON] = 0.5
            split_scale[SPLIT_ON:] = 0.5
            dividend[DIVIDEND_ON] = 0.5
            last_close = clean[DIVIDEND_ON - 1] * split_scale[DIVIDEND_ON - 1]
            dividend_scale[DIVIDEND_ON:] = 1 - dividend[DIVIDEND_ON] / last_close
        elif asset == "NANSPLIT":
            split_ratio[NANSPLIT_ON] = 0.5
            split_scale[NANSPLIT_ON:] = 0.5
        close = clean * split_scale * dividend_scale
        open_ = close * (1 + rng.normal(0, 0.005, bars))
        df = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.01,
                "low": np.minimum(open_, close) * 0.99,
                "close": close,
                "volume": rng.integers(1000, 2000, bars) / split_scale,
                "ex-dividend": dividend,
                "split_ratio": split_ratio,
                "clean_close": clean,
                "sector": sector,
            },
            index=DIRTY_DATES,
        )
        prices = ["open", "high", "low", "close", "clean_close"]
        if asset == "LISTED":
            df = df.iloc[LISTED_ON:]
        elif asset == "DELISTED":
            df = df.iloc[: DELISTED_ON + 1]
        elif asset == "HALTED":
            df = df.drop(DIRTY_DATES[HALTED_BARS])
        elif asset == "NANBAR":
            df.loc[DIRTY_DATES[NAN_BAR], prices] = np.nan
        elif asset == "NANSPLIT":
            df.loc[DIRTY_DATES[NANSPLIT_ON - 1], prices] = np.nan
        frames[asset] = df
    return pd.concat(frames, names=["asset", "date"]).swaplevel().sort_index()


@pytest.fixture(scope="session")
def dirty_panel():
    """The raw data of `dirty_loader`, for computing expected values with pandas."""
    return make_dirty_panel()


@pytest.fixture(scope="session")
def dirty_loader(dirty_panel):
    return spectre.data.MemoryLoader(dirty_panel, DIRTY_OHLCV, DIRTY_ADJUSTMENTS)


@pytest.fixture
def dirty_engine(dirty_loader, device):
    """An engine on the dirty panel, the test runs on cpu and on cuda if available."""
    engine = spectre.factors.FactorEngine(dirty_loader)
    if device == "cuda":
        engine.to_cuda()
    return engine


# --------------------------------- external data ---------------------------------


def external_data(file_name):
    """Path of the external data file, skips the test if not found."""
    path = EXTERNAL_DATA_DIR / file_name
    if not path.exists():
        pytest.skip(
            f"external data {path} not found, set SPECTRE_EXTERNAL_DATA to the folder contains it"
        )
    return path
