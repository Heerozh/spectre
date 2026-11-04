import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import spectre
from spectre import factors

OHLCV = ("open", "high", "low", "close", "volume")

pytestmark = pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")


@pytest.fixture
def shown(monkeypatch):
    """The figures that would have opened in a browser."""
    figures = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: figures.append(self))
    return figures


def make_panel(dates, assets=6, seed=0):
    """Assets that drift in the order of their `signal`, which is known at the open."""
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(assets)]
    shape = len(dates), assets
    quality = np.linspace(-1, 1, assets)
    returns = 0.002 * quality + rng.normal(0, 0.012, shape)
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=names)
    signal = pd.DataFrame(quality + rng.normal(0, 0.05, shape), index=dates, columns=names)
    panel = pd.DataFrame({"close": close.stack(), "signal": signal.stack()})
    for column in ("open", "high", "low"):
        panel[column] = panel["close"]
    panel["volume"] = 1e6
    return panel.rename_axis(["date", "asset"])


def chart(panel, shown, periods, timezone="UTC"):
    """Runs full_run with the preview on. Returns its data, and the lines of the real figure."""
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, ohlcv=OHLCV))
    engine.timezone = timezone
    engine.add(factors.ColumnDataFactor(inputs=("signal",), should_delay=False), "alpha")
    dates = panel.index.levels[0]
    factor_data, _ = engine.full_run(dates[0], dates[-1], periods=periods, quantiles=2)
    return factor_data, {t.name: t for t in shown[-1].data if t.type == "scatter"}


def held_for(panel, factor_data, bars):
    """The return of holding the weights of each bar for `bars` bars, from the raw prices."""
    weights = factor_data[("alpha", "factor_weight")].unstack()
    weights.index = weights.index.tz_convert("UTC")
    close = panel["close"].unstack()
    forward = (close.shift(-bars) / close - 1).reindex(index=weights.index, columns=weights.columns)
    return (weights * forward).sum(axis=1)


def cumulative_percent(returns):
    return ((1 + returns).cumprod().to_numpy() - 1) * 100


def expected_line(panel, factor_data, bars):
    """Every `bars` bars give one return, the mean of the portfolios started on those bars."""
    portfolio = held_for(panel, factor_data, bars)
    return cumulative_percent(portfolio.groupby(np.arange(len(portfolio)) // bars).mean())


def test_full_run_chart_compounds_the_returns(shown):
    panel = make_panel(pd.bdate_range("2018-01-01", periods=300, tz="UTC"))

    factor_data, lines = chart(panel, shown, (1, 4, 9))

    assert list(lines) == ["1D", "4D", "9D"]
    for bars in (1, 4, 9):
        expected = expected_line(panel, factor_data, bars)
        assert np.asarray(lines[f"{bars}D"].y) == pytest.approx(expected, rel=1e-3)
        # the drift of the panel is strong, so a lost bar can't go unnoticed
        assert expected[-1] > 40


def test_full_run_chart_points_are_the_starts_of_the_periods(shown):
    dates = pd.bdate_range("2018-01-01", periods=120, tz="UTC")

    factor_data, lines = chart(make_panel(dates), shown, (1, 4))

    bars = factor_data.index.get_level_values(0).unique()
    assert pd.DatetimeIndex(lines["1D"].x).equals(bars.tz_localize(None))
    assert pd.DatetimeIndex(lines["4D"].x).equals(bars[::4].tz_localize(None))


def test_full_run_chart_on_a_calendar_with_holidays(shown):
    dates = pd.bdate_range("2018-01-01", periods=300, tz="UTC").delete([7, 30, 31, 61, 200])
    panel = make_panel(dates)

    factor_data, lines = chart(panel, shown, (1, 4))

    assert np.asarray(lines["1D"].y) == pytest.approx(
        expected_line(panel, factor_data, 1), rel=1e-3
    )


@pytest.mark.parametrize(
    "timezone", ["UTC", "America/New_York", "Asia/Shanghai", "Asia/Tokyo", "Europe/London"]
)
def test_full_run_chart_in_a_timezone(shown, timezone, capsys):
    panel = make_panel(pd.bdate_range("2018-01-01", periods=200, tz="UTC"))

    factor_data, lines = chart(panel, shown, (1, 4, 9), timezone)

    assert np.asarray(lines["1D"].y) == pytest.approx(
        expected_line(panel, factor_data, 1), rel=1e-3
    )
    assert "re-sampling failed" not in capsys.readouterr().out


def test_full_run_chart_when_a_bin_starts_on_a_skipped_midnight(shown):
    # 2019-03-22 00:00 doesn't exist in Tehran, and it is 5 business days after the last bar
    dates = pd.bdate_range("2019-02-25", "2019-03-25", tz="UTC")
    panel = make_panel(dates)

    factor_data, lines = chart(panel, shown, (1, 5), "Asia/Tehran")

    assert str(factor_data.index.get_level_values(0)[-1].date()) == "2019-03-15"
    assert np.asarray(lines["1D"].y) == pytest.approx(
        expected_line(panel, factor_data, 1), rel=1e-3
    )


@pytest.mark.filterwarnings("error::pandas.errors.Pandas4Warning")
def test_full_run_chart_has_no_deprecated_frequency(shown):
    chart(make_panel(pd.bdate_range("2018-01-01", periods=60, tz="UTC")), shown, (1, 4))


@pytest.mark.parametrize(
    "bar, periods, names",
    [
        ("5min", (12, 1, 4), ["5min", "20min", "60min"]),
        ("1h", (1, 4, 24), ["1h", "4h", "24h"]),
        ("30s", (1, 2, 10), ["30s", "60s", "300s"]),
        ("250ms", (1, 4), ["250ms", "1000ms"]),
    ],
)
def test_full_run_chart_of_intraday_bars(shown, bar, periods, names):
    panel = make_panel(pd.date_range("2019-01-07", periods=400, freq=bar, tz="UTC"))

    factor_data, lines = chart(panel, shown, periods)

    # in order of length, a sort by name would put 20min first
    assert list(lines) == names
    for name, bars in zip(names, sorted(periods)):
        # these bins follow the clock and not the first bar
        portfolio = held_for(panel, factor_data, bars)
        expected = cumulative_percent(portfolio.groupby(portfolio.index.floor(name)).mean())
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_full_run_chart_of_the_5_minute_test_data(shown, intraday_loader):
    engine = factors.FactorEngine(intraday_loader)
    engine.add(factors.MA(3), "alpha")

    factor_data, _ = engine.full_run("2019-01-03", "2019-01-11", periods=(1, 4, 12), quantiles=2)

    lines = {t.name: t for t in shown[-1].data if t.type == "scatter"}
    assert list(lines) == ["5min", "20min", "60min"]
    # one point for each bar of the 5 minutes line, the nights have no bars
    assert len(lines["5min"].y) == factor_data.index.get_level_values(0).nunique()
    assert all(np.isfinite(line.y).all() for line in lines.values())
