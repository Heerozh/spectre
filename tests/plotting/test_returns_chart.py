import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import spectre
from spectre import factors
from spectre.plotting import plot_quantile_and_cumulative_returns

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


def alpha_engine(panel, timezone="UTC", universe=None):
    engine = factors.FactorEngine(spectre.data.MemoryLoader(panel, ohlcv=OHLCV))
    engine.timezone = timezone
    engine.add(factors.ColumnDataFactor(inputs=("signal",), should_delay=False), "alpha")
    if universe is not None:
        engine.set_filter(universe)
    return engine


def lines_of(figure):
    return {trace.name: trace for trace in figure.data if trace.type == "scatter"}


def chart(panel, shown, periods, timezone="UTC", universe=None, start=None):
    """Runs full_run with the preview on. Returns its data, and the lines of the real figure."""
    dates = panel.index.levels[0]
    engine = alpha_engine(panel, timezone, universe)
    start = dates[0] if start is None else start
    factor_data, _ = engine.full_run(start, dates[-1], periods=periods, quantiles=2)
    return factor_data, lines_of(shown[-1])


def filtered_to(panel, bars):
    """The panel with a column for a filter that leaves `bars`, positions in its dates."""
    dates = panel.index.get_level_values(0)
    panel = panel.assign(kept=dates.isin(panel.index.levels[0][bars]).astype(float))
    return panel, factors.ColumnDataFactor(inputs=("kept",)) > 0.5


def held_for(panel, factor_data, bars):
    """The return of holding the weights of each bar for `bars` bars, from the raw prices."""
    weights = factor_data[("alpha", "factor_weight")].unstack()
    weights.index = weights.index.tz_convert("UTC")
    close = panel["close"].unstack()
    forward = (close.shift(-bars) / close - 1).reindex(index=weights.index, columns=weights.columns)
    return (weights * forward).sum(axis=1)


def cumulative_percent(returns):
    return ((1 + returns).cumprod().to_numpy() - 1) * 100


def expected_bins(panel, rows, bars):
    """Rows by the `bars` bars of the panel they fall in, counted from the first row."""
    position = panel.index.levels[0].get_indexer(rows)
    return (position - position[0]) // bars


def expected_line(panel, factor_data, bars):
    """One return per `bars` bars of the panel, the mean of the portfolios started on them."""
    portfolio = held_for(panel, factor_data, bars)
    return cumulative_percent(portfolio.groupby(expected_bins(panel, portfolio.index, bars)).mean())


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

    factor_data, lines = chart(make_panel(dates), shown, (1, 4), "Asia/Tokyo")

    # timezone aware, like the turnover bars and the zero line on the same axes
    bars = factor_data.index.get_level_values(0).unique()
    assert pd.DatetimeIndex(lines["1D"].x).equals(bars)
    assert pd.DatetimeIndex(lines["4D"].x).equals(bars[::4])
    turnover = next(trace for trace in shown[-1].data if trace.name == "turnover")
    assert pd.DatetimeIndex(turnover.x).tz is not None
    assert pd.DatetimeIndex(lines["1D"].x).tz is not None


def weekdays_and_holidays():
    return pd.bdate_range("2018-01-01", periods=300, tz="UTC").delete([7, 30, 31, 61, 200])


def weekly_bars():
    return pd.date_range("2015-01-02", periods=200, freq="7D", tz="UTC")


def seven_days_a_week():
    return pd.date_range("2018-01-01", periods=300, freq="D", tz="UTC")


def monthly_bars():
    return pd.date_range("2005-01-01", periods=100, freq="MS", tz="UTC")


def two_bars_a_day():
    days = pd.bdate_range("2018-01-01", periods=150, tz="UTC")
    return days.repeat(2) + pd.to_timedelta(np.tile([9.5, 13.0], len(days)), unit="h")


@pytest.mark.parametrize(
    "bars",
    [weekdays_and_holidays, weekly_bars, seven_days_a_week, monthly_bars, two_bars_a_day],
)
def test_full_run_chart_on_any_calendar(shown, bars):
    panel = make_panel(bars())

    factor_data, lines = chart(panel, shown, (1, 4, 9))

    for name, count in zip(sorted(lines, key=pd.Timedelta), (1, 4, 9)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_full_run_chart_of_bars_on_a_local_saturday(shown):
    # the bars of Friday evening in New York are on Saturday morning in Tokyo
    panel = make_panel(pd.date_range("2018-01-01 20:00", periods=300, freq="B", tz="UTC"))

    factor_data, lines = chart(panel, shown, (1, 4, 9), "Asia/Tokyo")

    assert (factor_data.index.get_level_values(0).day_of_week == 5).any()
    for name, count in zip(sorted(lines, key=pd.Timedelta), (1, 4, 9)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "timezone", ["UTC", "America/New_York", "Asia/Shanghai", "Asia/Tokyo", "Europe/London"]
)
def test_full_run_chart_in_a_timezone(shown, timezone):
    panel = make_panel(pd.bdate_range("2018-01-01", periods=200, tz="UTC"))

    factor_data, lines = chart(panel, shown, (1, 4, 9), timezone)

    for name, count in zip(sorted(lines, key=pd.Timedelta), (1, 4, 9)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "first, last, last_row",
    [
        # a rule by the calendar has its last bin edge on the midnight
        ("2019-02-25", "2019-03-22", "2019-03-15"),
        # the rows go over the midnight
        ("2019-03-11", "2019-04-12", "2019-04-05"),
        # and one of the bars is at that moment, Friday evening is Saturday there
        ("2019-03-11 20:30", "2019-04-12 20:30", "2019-04-06"),
    ],
)
def test_full_run_chart_around_a_midnight_skipped_by_daylight_saving(shown, first, last, last_row):
    # 2019-03-22 00:00 doesn't exist in Tehran
    panel = make_panel(pd.date_range(first, last, freq="B", tz="UTC"))

    factor_data, lines = chart(panel, shown, (1, 5), "Asia/Tehran")

    assert str(factor_data.index.get_level_values(0)[-1].date()) == last_row
    for name, count in (("1D", 1), ("5D", 5)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_full_run_chart_counts_bins_from_its_first_row(shown):
    """The loaded dates start earlier, for the windows, and that must not cut a bin short."""
    dates = pd.bdate_range("2018-01-01", periods=300, tz="UTC")
    panel = make_panel(dates)

    factor_data, lines = chart(panel, shown, (1, 4, 9), start=dates[30])

    bars = factor_data.index.get_level_values(0).unique()
    assert pd.DatetimeIndex(lines["4D"].x).equals(bars[::4])
    assert pd.DatetimeIndex(lines["9D"].x).equals(bars[::9])
    for name, count in (("1D", 1), ("4D", 4), ("9D", 9)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def every_2nd_bar(dates):
    return np.arange(0, len(dates), 2)


def every_5th_bar(dates):
    return np.arange(0, len(dates), 5)


def mondays(dates):
    return np.flatnonzero(dates.day_of_week == 0)


def first_bar_of_the_month(dates):
    bars = pd.Series(np.arange(len(dates)), index=dates)
    return bars.groupby([dates.year, dates.month]).first().to_numpy()


def random_20_percent(dates):
    return np.sort(np.random.default_rng(0).choice(len(dates), len(dates) // 5, replace=False))


def random_half(dates):
    return np.sort(np.random.default_rng(0).choice(len(dates), len(dates) // 2, replace=False))


def bursts(dates):
    return np.concatenate([np.arange(start, start + 3) for start in range(0, len(dates) - 3, 13)])


FILTERS = [
    every_2nd_bar,
    every_5th_bar,
    mondays,
    first_bar_of_the_month,
    random_20_percent,
    random_half,
    bursts,
]


@pytest.mark.parametrize("kept", FILTERS)
def test_full_run_chart_counts_the_bars_a_filter_removed(shown, kept):
    dates = pd.bdate_range("2018-01-01", periods=400, tz="UTC")
    panel, universe = filtered_to(make_panel(dates), kept(dates))

    factor_data, lines = chart(panel, shown, (1, 4, 9), universe=universe)

    for name, count in (("1D", 1), ("4D", 4), ("9D", 9)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_full_run_chart_points_with_a_filter_are_the_first_rows_of_the_bins(shown):
    dates = pd.bdate_range("2018-01-01", periods=400, tz="UTC")
    panel, universe = filtered_to(make_panel(dates), random_20_percent(dates))

    factor_data, lines = chart(panel, shown, (1, 4, 9), universe=universe)

    rows = factor_data.index.get_level_values(0).unique()
    for name, count in (("1D", 1), ("4D", 4), ("9D", 9)):
        first_rows = ~pd.Index(expected_bins(panel, rows, count)).duplicated()
        assert pd.DatetimeIndex(lines[name].x).equals(rows[first_rows])


def test_full_run_chart_of_a_single_row(shown):
    dates = pd.bdate_range("2018-01-01", periods=30, tz="UTC")
    panel, universe = filtered_to(make_panel(dates), [10])

    factor_data, lines = chart(panel, shown, (1, 4), universe=universe)

    assert factor_data.index.get_level_values(0).nunique() == 1
    assert len(lines["1D"].y) == len(lines["4D"].y) == 1


def preview_data(panel, universe=None):
    dates = panel.index.levels[0]
    return alpha_engine(panel, universe=universe).full_run(
        dates[0], dates[-1], periods=(1, 4), quantiles=2, preview=False
    )


def test_chart_without_dates_goes_by_the_rows():
    panel = make_panel(pd.bdate_range("2018-01-01", periods=120, tz="UTC"))
    factor_data, mean_returns = preview_data(panel)

    lines = lines_of(plot_quantile_and_cumulative_returns(factor_data, mean_returns))

    for name, count in (("1D", 1), ("4D", 4)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_chart_given_the_dates_counts_the_bars_a_filter_removed():
    dates = pd.bdate_range("2018-01-01", periods=200, tz="UTC")
    panel, universe = filtered_to(make_panel(dates), random_20_percent(dates))
    factor_data, mean_returns = preview_data(panel, universe)

    figure = plot_quantile_and_cumulative_returns(factor_data, mean_returns, dates)

    lines = lines_of(figure)
    for name, count in (("1D", 1), ("4D", 4)):
        expected = expected_line(panel, factor_data, count)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_chart_without_dates_of_rows_further_apart_than_a_period():
    """The rows are all it has, a period shorter than their gap must not make an empty bin."""
    dates = pd.bdate_range("2018-01-01", periods=200, tz="UTC")
    panel, universe = filtered_to(make_panel(dates), every_5th_bar(dates))
    factor_data, mean_returns = preview_data(panel, universe)

    lines = lines_of(plot_quantile_and_cumulative_returns(factor_data, mean_returns))

    rows = factor_data.index.get_level_values(0).nunique()
    assert len(lines["1D"].y) == len(lines["4D"].y) == rows


def test_chart_without_dates_of_a_single_row():
    dates = pd.bdate_range("2018-01-01", periods=30, tz="UTC")
    panel, universe = filtered_to(make_panel(dates), [10])
    factor_data, mean_returns = preview_data(panel, universe)

    lines = lines_of(plot_quantile_and_cumulative_returns(factor_data, mean_returns))

    assert len(lines["1D"].y) == len(lines["4D"].y) == 1


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
        expected = expected_line(panel, factor_data, bars)
        assert np.asarray(lines[name].y) == pytest.approx(expected, rel=1e-3)


def test_full_run_chart_of_the_5_minute_test_data(shown, intraday_loader):
    engine = factors.FactorEngine(intraday_loader)
    engine.add(factors.MA(3), "alpha")

    factor_data, _ = engine.full_run("2019-01-03", "2019-01-11", periods=(1, 4, 12), quantiles=2)

    lines = lines_of(shown[-1])
    assert list(lines) == ["5min", "20min", "60min"]
    # one point for each bar of the 5 minutes line, the nights have no bars
    assert len(lines["5min"].y) == factor_data.index.get_level_values(0).nunique()
    assert all(np.isfinite(line.y).all() for line in lines.values())
