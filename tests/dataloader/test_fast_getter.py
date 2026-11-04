import numpy as np
import pandas as pd
import pytest

import spectre
from tests.conftest import DATA_DIR

DAY = pd.Timestamp("2018-01-02", tz="UTC")


@pytest.fixture(scope="module")
def prices():
    loader = spectre.data.CsvDirLoader(
        str(DATA_DIR / "daily"), prices_index="date", parse_dates=True
    )
    return loader.load()[list(loader.ohlcv)]


@pytest.fixture
def getter(prices):
    return spectre.data.DataLoaderFastGetter(prices)


def test_get_as_dict_column(prices, getter):
    table = getter.get_as_dict(DAY, column_id=3)
    assert table["MSFT"] == prices.loc[("2018-01-02", "MSFT")].close
    assert table["AAPL"] == prices.loc[("2018-01-02", "AAPL")].close
    with pytest.raises(KeyError):
        table["A"]
    assert table.get("A", 1.0) == 1.0

    table = dict(table.items())
    assert table == {
        "AAPL": prices.loc[("2018-01-02", "AAPL")].close,
        "MSFT": prices.loc[("2018-01-02", "MSFT")].close,
    }


def test_get_as_dict_row(prices, getter):
    table = getter.get_as_dict(DAY)
    np.testing.assert_array_equal(table["MSFT"], prices.loc[("2018-01-02", "MSFT")].values)
    np.testing.assert_array_equal(table["AAPL"], prices.loc[("2018-01-02", "AAPL")].values)


def test_get_as_df(prices, getter):
    result = getter.get_as_df(DAY)
    pd.testing.assert_frame_equal(prices.xs("2018-01-02"), result)


def test_missing_data(getter):
    # MSFT has a junk row on 2019-01-05, AAPL has no data
    table = getter.get_as_dict(pd.Timestamp("2019-01-05", tz="UTC"), column_id=3)
    assert np.isnan(table["MSFT"])
    with pytest.raises(KeyError):
        table["AAPL"]

    # MSFT has no data on 2019-01-10
    table = getter.get_as_dict(pd.Timestamp("2019-01-10", tz="UTC"), column_id=3)
    with pytest.raises(KeyError):
        table["MSFT"]
    assert "AAPL" in dict(table.items())


def test_get_range(intraday_loader):
    df = intraday_loader.load()
    getter = spectre.data.DataLoaderFastGetter(df)
    table = getter.get_as_dict(
        pd.Timestamp("2018-12-20 00:00:00", tz="UTC"),
        pd.Timestamp("2018-12-20 23:59:59", tz="UTC"),
        column_id=0,
    )
    dates = table.get_datetime_index()
    assert len(dates.normalize().unique()) == 1
    assert len(dates) == len(df.loc["2018-12-20"])
