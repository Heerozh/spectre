"""
Shared fixtures for the spectre test-suite.

Run all tests in parallel:  ``pytest -n auto``  (needs pytest-xdist)
"""
import os
from pathlib import Path

import pytest
import torch

import spectre

DATA_DIR = Path(__file__).parent / 'data'

# The test data is a slice of IEX data of AAPL and MSFT, with some deliberate defects:
#   * MSFT has an all-NaN row on 2019-01-05 (Saturday), it is dropped when the loader uses
#     `calender_asset='AAPL'`.
#   * MSFT has no data on 2019-01-10.
DAILY_OHLCV = ('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume')
INTRADAY_OHLCV = ('Open', 'High', 'Low', 'Close', 'Volume')

# Datasets which are too big to be included in this repository, tests using them are skipped
# when they are not found. Defaults to `../historical_data/us/prices/quandl` next to the repo.
EXTERNAL_DATA_DIR = Path(os.environ.get(
    'SPECTRE_EXTERNAL_DATA',
    Path(__file__).parents[2] / 'historical_data' / 'us' / 'prices' / 'quandl'))


def pytest_addoption(parser):
    parser.addoption('--run-network', action='store_true', default=False,
                     help='run tests that download data from the internet')


def pytest_collection_modifyitems(config, items):
    skip_cuda = pytest.mark.skip(reason='CUDA is not available')
    skip_network = pytest.mark.skip(reason='needs internet, use --run-network to run')
    has_cuda = torch.cuda.is_available()
    run_network = config.getoption('--run-network')
    for item in items:
        if 'cuda' in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if 'network' in item.keywords and not run_network:
            item.add_marker(skip_network)


# ------------------------------------ devices ------------------------------------


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def device(request):
    """Run the test on cpu, and on cuda if available."""
    return request.param


# ------------------------------------ loaders ------------------------------------
# Loaders don't cache anything, so they can be safely shared within a test session.


def make_daily_loader(**kwargs):
    params = dict(ohlcv=DAILY_OHLCV, prices_index='date', parse_dates=True)
    params.update(kwargs)
    return spectre.data.CsvDirLoader(str(DATA_DIR / 'daily'), **params)


@pytest.fixture(scope='session')
def daily_loader():
    """Daily prices of AAPL and MSFT, aligned to the trading calendar of AAPL."""
    return make_daily_loader(calender_asset='AAPL')


@pytest.fixture(scope='session')
def daily_loader_no_calendar():
    """Daily prices, without calendar alignment, so the junk row of MSFT (2019-01-05) is kept."""
    return make_daily_loader()


def make_adjusted_daily_loader(**kwargs):
    return make_daily_loader(
        dividends_path=str(DATA_DIR / 'dividends'), splits_path=str(DATA_DIR / 'splits'),
        adjustments=('amount', 'ratio'), dividends_index='exDate', splits_index='exDate',
        **kwargs)


@pytest.fixture(scope='session')
def adjusted_daily_loader():
    """Daily prices with dividends and splits (MSFT 1-for-15 reverse split on 2019-01-14)."""
    return make_adjusted_daily_loader(calender_asset='AAPL')


@pytest.fixture(scope='session')
def adjusted_daily_loader_no_calendar():
    return make_adjusted_daily_loader()


@pytest.fixture(scope='session')
def intraday_loader():
    """5 minutes bars of AAPL and MSFT, from 2018-12 to 2019-01."""
    return spectre.data.CsvDirLoader(
        str(DATA_DIR / '5mins'), prices_by_year=True, prices_index='Date',
        ohlcv=INTRADAY_OHLCV, parse_dates=True)


# ------------------------------------ engines ------------------------------------


@pytest.fixture
def engine(daily_loader):
    return spectre.factors.FactorEngine(daily_loader)


@pytest.fixture
def engine_no_calendar(daily_loader_no_calendar):
    return spectre.factors.FactorEngine(daily_loader_no_calendar)


# --------------------------------- external data ---------------------------------


@pytest.fixture(scope='session')
def quandl_dir():
    if not (EXTERNAL_DATA_DIR / 'wiki_prices.feather').exists():
        pytest.skip('external data {} not found, set SPECTRE_EXTERNAL_DATA to the folder '
                    'contains wiki_prices.feather'.format(EXTERNAL_DATA_DIR))
    return EXTERNAL_DATA_DIR
