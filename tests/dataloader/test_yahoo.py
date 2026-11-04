"""
YahooDownloader without the network, `test_external.py` has the real one.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from spectre import factors
from spectre.data import ArrowLoader, YahooDownloader, yahoo
from spectre.data.yahoo import _to_raw_prices

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
DATES = pd.bdate_range("2020-01-06", periods=6, name="Date")


def yahoo_bars(closes, splits=None, dividends=None):
    """Bars as yfinance returns them with `auto_adjust=False`, already split adjusted."""
    closes = np.array(closes, dtype=float)
    empty = np.zeros(len(closes))
    bars = {
        "Open": closes,
        "High": closes + 1,
        "Low": closes - 1,
        "Close": closes,
        "Adj Close": closes,
        "Volume": 1000.0,
        "Dividends": empty if dividends is None else dividends,
        "Stock Splits": empty if splits is None else splits,
    }
    return pd.DataFrame(bars, index=DATES[: len(closes)])


def test_undo_split_adjustment():
    # a 2-for-1 split on the 4th bar, yahoo halves the prices and the dividend before it
    bars = yahoo_bars(
        [50, 51, 50, 52, 53, 54], splits=[0, 0, 0, 2, 0, 0], dividends=[0, 0, 0.5, 0, 0, 0]
    )
    raw = _to_raw_prices(bars)
    assert list(raw["Close"]) == [100, 102, 100, 52, 53, 54]
    assert list(raw["High"]) == [102, 104, 102, 53, 54, 55]
    assert list(raw["Volume"]) == [500, 500, 500, 1000, 1000, 1000]
    assert list(raw["ex-dividend"]) == [0, 0, 1, 0, 0, 0]
    assert list(raw["split_ratio"]) == [1, 1, 1, 0.5, 1, 1]


def test_several_splits_and_reverse_splits():
    raw = _to_raw_prices(yahoo_bars([10, 10, 10, 10], splits=[0, 2, 0.5, 3]))
    assert list(raw["Close"]) == [30, 15, 30, 10]
    assert list(raw["split_ratio"]) == [1, 0.5, 2, 1 / 3]


def test_dividend_on_a_split_date():
    # it's paid on the shares before the split, so it has to match the close of the day before
    raw = _to_raw_prices(yahoo_bars([50, 52, 53], splits=[0, 2, 0], dividends=[0, 0.5, 0]))
    assert list(raw["Close"]) == [100, 52, 53]
    assert list(raw["ex-dividend"]) == [0, 1, 0]


@pytest.fixture
def fake_yahoo(monkeypatch):
    """Replace `yfinance.download`, and return the list of its calls."""
    plain = yahoo_bars([10, 11, 12, 13, 14, 15])
    bars = {
        "AAA": yahoo_bars(
            [50, 51, 50, 52, 53, 54], splits=[0, 0, 0, 2, 0, 0], dividends=[0, 0, 0.5, 0, 0, 0]
        ),
        "BBB": plain,
        "BRK-B": plain,
        "SPY": plain,
        "QQQ": plain,
    }
    no_data = pd.DataFrame(np.nan, index=DATES, columns=COLUMNS)
    calls = []

    def download(tickers, **kwargs):
        calls.append((list(tickers), kwargs))
        frames = {ticker: bars.get(ticker, no_data) for ticker in tickers}
        return pd.concat(frames, axis=1, names=["Ticker", "Price"])

    monkeypatch.setattr("yfinance.download", download)
    monkeypatch.setattr(YahooDownloader, "retry_delays", (0,))
    return calls


@pytest.mark.filterwarnings("ignore:Warning!! delay_factor is set to False")
def test_ingest_enables_adjustments(fake_yahoo, tmp_path):
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA", "BBB"])
    loader = ArrowLoader(str(tmp_path / "yahoo.feather"))
    assert loader.adjustments == ("ex-dividend", "split_ratio")

    df = loader.load()
    aaa = df.xs("AAA", level="asset")
    assert_allclose(aaa["Close"], [100, 102, 100, 52, 53, 54])
    # the ex-dividend and split moved up one row, onto the last bar before them
    assert_allclose(aaa["ex-dividend"], [0, 1, 0, 0, 0, 0])
    assert_allclose(aaa["split_ratio"], [1, 1, 0.5, 1, 1, 1])
    dividend = 1 - 1 / 102
    assert_allclose(aaa["price_multi"], [dividend / 2, dividend / 2, 0.5, 1, 1, 1], rtol=1e-6)
    assert_allclose(aaa["vol_multi"], [2, 2, 2, 1, 1, 1])

    engine = factors.FactorEngine(loader)
    engine.add(factors.Returns(), "returns")
    returns = engine.run("2020-01-07", "2020-01-13", delay_factor=False)
    returns = returns.xs("AAA", level="asset")["returns"]
    # continuous over the split, and with the dividend added back
    expected = [51 / 50 - 1, 100 / 101 - 1, 52 / 50 - 1, 53 / 52 - 1, 54 / 53 - 1]
    assert_allclose(returns, expected, rtol=1e-5)


def test_ingest_asks_for_unadjusted_finished_bars(fake_yahoo, tmp_path):
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA"])
    _, kwargs = fake_yahoo[0]
    assert kwargs["auto_adjust"] is False
    assert kwargs["actions"] is True
    # `end` is exclusive, so it can't go past today in the timezone of the market
    assert pd.Timestamp(kwargs["end"]) <= pd.Timestamp.now(tz="America/New_York").tz_localize(None)


def test_ingest_retries_and_reports_symbols_without_data(fake_yahoo, tmp_path, capsys):
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA", "BAD"])
    assert [call[0] for call in fake_yahoo] == [["AAA", "BAD"], ["BAD"]]
    assert "skipped: ['BAD']" in capsys.readouterr().out
    assert not (tmp_path / "daily" / "BAD.csv").exists()
    assert list(ArrowLoader(str(tmp_path / "yahoo.feather")).load().index.levels[1]) == ["AAA"]


def test_ingest_downloads_in_batches(fake_yahoo, tmp_path, monkeypatch):
    monkeypatch.setattr(YahooDownloader, "batch_size", 2)
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA", "BBB", "SPY"])
    assert [call[0] for call in fake_yahoo] == [["AAA", "BBB"], ["SPY"]]


def test_ingest_skip_exists(fake_yahoo, tmp_path):
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA"])
    fake_yahoo.clear()
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA", "BBB"])
    assert [call[0] for call in fake_yahoo] == [["BBB"]]
    fake_yahoo.clear()
    YahooDownloader.ingest("2020", str(tmp_path), ["AAA", "BBB"], skip_exists=False)
    assert [call[0] for call in fake_yahoo] == [["AAA", "BBB"]]


def test_ingest_uses_yahoo_symbols(fake_yahoo, tmp_path):
    YahooDownloader.ingest("2020", str(tmp_path), ["brk.b"])
    assert fake_yahoo[0][0] == ["BRK-B"]
    assert (tmp_path / "daily" / "BRK-B.csv").exists()


@pytest.mark.filterwarnings("ignore:`symbols=None` downloads")
def test_split_on_a_day_without_a_bar_of_the_calendar(tmp_path, monkeypatch):
    calendar = yahoo_bars([10, 11, 12, 13, 14, 15]).drop(DATES[3])
    bars = {
        "AAA": yahoo_bars([50, 51, 50, 52, 53, 54], splits=[0, 0, 0, 2, 0, 0]),
        "SPY": calendar,
        "QQQ": calendar,
    }

    def download(tickers, **kwargs):
        frames = {ticker: bars[ticker] for ticker in tickers}
        return pd.concat(frames, axis=1, names=["Ticker", "Price"])

    members = pd.Series({pd.Timestamp("2020-01-02"): frozenset({"AAA"})})
    monkeypatch.setattr("yfinance.download", download)
    monkeypatch.setattr(yahoo, "sp500_constituents", lambda: members)
    YahooDownloader.ingest("2020", str(tmp_path))

    aaa = ArrowLoader(str(tmp_path / "yahoo.feather")).load().xs("AAA", level="asset")
    # the bar of the ex-date is dropped by the calendar, the split moved to the next bar
    assert_allclose(aaa["Close"], [100, 102, 100, 53, 54])
    assert_allclose(aaa["split_ratio"], [1, 1, 0.5, 1, 1])
    assert_allclose(aaa["price_multi"], [0.5, 0.5, 0.5, 1, 1])
    assert_allclose(aaa["vol_multi"], [2, 2, 2, 1, 1])


def test_default_symbols_are_the_latest_members(fake_yahoo, tmp_path, monkeypatch):
    members = pd.Series(
        {
            pd.Timestamp("2019-01-02"): frozenset({"DELISTED"}),
            pd.Timestamp("2020-01-02"): frozenset({"AAA", "BRK.B"}),
        }
    )
    monkeypatch.setattr(yahoo, "sp500_constituents", lambda: members)
    with pytest.warns(RuntimeWarning, match="survivorship bias"):
        YahooDownloader.ingest("2020", str(tmp_path))
    assert fake_yahoo[0][0] == ["AAA", "BRK-B", "SPY", "QQQ"]
