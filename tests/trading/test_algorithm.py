import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from numpy import nan
from numpy.testing import assert_almost_equal, assert_array_equal

import spectre
from spectre import factors, trading


def ts(date):
    return pd.Timestamp(date, tz="UTC")


def live_alg(periods, **sources):
    dates = pd.bdate_range("2019-01-01", periods=periods, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    ohlcv = ("open", "high", "low", "close", "volume")
    panel = pd.DataFrame(100.0, index=index, columns=ohlcv)
    loader = spectre.data.MemoryLoader(panel, ohlcv=ohlcv)
    alg = trading.CustomAlgorithm(trading.SimulationBlotter(loader), main=loader)
    for name in sources:
        alg.create_factor_engine(name, loader)
    alg.get_factor_engine().add(factors.OHLCV.close, "close")
    for name in sources:
        alg.get_factor_engine(name).add(factors.OHLCV.close, "close")
    alg.set_datetime(dates[-1] + pd.Timedelta(days=1))
    return alg, dates


@pytest.mark.parametrize("periods", [300, 100], ids=["over_a_year", "under_a_year"])
@pytest.mark.parametrize("delay_factor", [True, False])
def test_run_engine_waits_for_the_next_bar(delay_factor, periods):
    """Live the clock runs ahead of the data, a bar that hasn't come yet is not an error."""
    alg, dates = live_alg(periods)

    data, last_row = alg.run_engine(None, None, delay_factor=delay_factor)

    assert len(data) > 0
    assert data.index.get_level_values(0)[-1] == dates[-1]
    assert len(last_row) == 0


@pytest.mark.parametrize("delay_factor", [True, False])
def test_run_engine_waits_for_the_next_bar_of_every_source(delay_factor):
    alg, dates = live_alg(100, other=True)

    data, last_rows = alg.run_engine(None, None, delay_factor=delay_factor)

    assert set(data) == {"main", "other"}
    for name in data:
        assert data[name].index.get_level_values(0)[-1] == dates[-1]
        assert len(last_rows[name]) == 0


@pytest.mark.filterwarnings("ignore:quantile.. mask parameter not set")
def test_market_open_earns_the_returns_of_full_run():
    """A delayed factor enters at the open of its row, in full_run and in a backtest."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=40, tz="UTC")
    assets = list("ABCD")
    shape = len(dates), len(assets)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, shape), axis=0))
    opens = pd.DataFrame(walk, index=dates, columns=assets)
    closes = opens * np.exp(rng.normal(0, 0.01, shape))
    columns = {
        "open": opens,
        "high": np.maximum(opens, closes),
        "low": np.minimum(opens, closes),
        "close": closes,
        "volume": opens * 0 + 1000,
        # known after a close, it tells the return from the open of the next bar
        "signal": (opens.shift(-1) / opens - 1).shift(-1),
    }
    panel = pd.concat({col: df.stack() for col, df in columns.items()}, axis=1)
    ohlcv = ("open", "high", "low", "close", "volume")
    loader = spectre.data.MemoryLoader(panel.rename_axis(["date", "asset"]), ohlcv=ohlcv)
    start, end = dates[5], dates[-3]

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            self.get_factor_engine().add(factors.ColumnDataFactor(inputs=("signal",)), "signal")
            self.blotter.set_commission(0, 0, 0)
            self.blotter.set_slippage(0, 0)
            self.schedule_rebalance(trading.event.MarketOpen(self.rebalance))

        def rebalance(self, data, history):
            weights = (data.index == data.signal.idxmax()).astype(float)
            self.blotter.batch_order_target_percent(data.index, weights)

    earned = (1 + trading.run_backtest(loader, Alg, start, end).returns.fillna(0)).prod()

    engine = factors.FactorEngine(loader)
    engine.add(factors.ColumnDataFactor(inputs=("signal",)), "signal")
    factor_data, _ = engine.full_run(
        start, dates[-1], trade_at="open", periods=(1,), quantiles=2, preview=False
    )
    returns = factor_data["Returns"]["1D"].unstack()

    top = panel["signal"].unstack().shift(1).loc[dates[6] : end].idxmax(axis=1)
    by_hand, one_bar_later, reported = 1.0, 1.0, 1.0
    for date, asset in top.iloc[:-1].items():
        bar = dates.get_loc(date)
        by_hand *= opens[asset].iloc[bar + 1] / opens[asset].iloc[bar]
        one_bar_later *= opens[asset].iloc[bar + 2] / opens[asset].iloc[bar + 1]
        reported *= 1 + returns.loc[date, asset]
    assert reported == pytest.approx(by_hand)
    # the backtest ends at the close of its last bar
    last_day = closes.loc[end, top.iloc[-1]] / opens.loc[end, top.iloc[-1]]
    assert earned == pytest.approx(by_hand * last_day, rel=0.01)
    assert by_hand > 2 * one_bar_later


def test_live_bar_reads_up_to_the_last_bar(daily_loader):
    """A live bar asks for the data from now on, it doesn't know the end of it."""

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            self.get_factor_engine().add(factors.MA(5), "ma5")

    alg = Alg(trading.SimulationBlotter(daily_loader), main=daily_loader)
    alg.initialize()
    alg.set_datetime(ts("2019-01-11"))
    alg._data_updated()
    assert alg._data.index.get_level_values(0)[-1] == daily_loader.load().index.levels[0][-1]


def test_simulation_event_manager(daily_loader_no_calendar):
    loader = daily_loader_no_calendar

    class MockAlgorithm(trading.CustomAlgorithm):
        def __init__(self):
            # don't call super().__init__, mock everything
            self.blotter = trading.SimulationBlotter(loader)
            self._event_manager = None
            self.data = None
            self.bar_dates = []  # elapsed time
            self.sequence = []  # sequence of events per day

        def clear(self):
            pass

        def run_engine(self, start, end, _=False):
            engine = factors.FactorEngine(loader)
            engine.add(factors.MA(5), "f")
            df = engine.run(start, end)
            return df, df.loc[df.index.get_level_values(0)[-1]]

        def on_run(self):
            def run_engine(_):
                self.data, _ = self.run_engine(None, None)

            self.schedule(trading.event.EveryBarData(run_engine))
            self.schedule(trading.event.EveryBarData(self.every_bar))
            self.schedule(trading.event.MarketOpen(self.event("before_open"), -1000))
            self.schedule(trading.event.MarketOpen(self.event("open"), 0))
            self.schedule(trading.event.MarketClose(self.event("before_close"), -1000))
            self.schedule(trading.event.MarketClose(self.event("close"), 1000))

        def on_end_of_run(self):
            pass

        def event(self, name):
            def callback(_):
                self.sequence.append(name)

            return callback

        def every_bar(self, _):
            self.sequence.append("bar")
            today = self.data.index.get_level_values(0)[-1]
            self.bar_dates.append(today)
            if today > ts("2019-01-10"):
                self.stop_event_manager()

    alg = MockAlgorithm()
    evt_mgr = trading.SimulationEventManager()
    evt_mgr.subscribe(alg)
    evt_mgr.run("2019-01-01", "2019-01-15")

    # delayed data starts from the second day
    assert alg.bar_dates[0] == ts("2019-01-03")
    assert alg.bar_dates[1] == ts("2019-01-04")
    # stopped
    assert alg.bar_dates[-1] == ts("2019-01-11")
    days = len(alg.bar_dates)
    assert alg.sequence == ["bar", "before_open", "open", "before_close", "close"] * days


@pytest.fixture
def one_engine_results(adjusted_daily_loader):
    class OneEngineAlg(trading.CustomAlgorithm):
        msft_ma5 = []

        def initialize(self):
            engine = self.get_factor_engine()
            ma5 = factors.MA(5)
            engine.add(ma5, "ma5")
            engine.set_filter(ma5.top(5))

            self.schedule_rebalance(trading.event.MarketOpen(self.rebalance))

            self.blotter.long_only = True
            self.blotter.set_commission(0, 0, 0)
            self.blotter.set_slippage(0, 0)

        def rebalance(self, data, history):
            # don't buy MSFT on the first day, MA5 of 01-04 to 01-11 is 103.9516
            if 103.95 <= data.loc["MSFT", "ma5"] <= 103.96:
                data = data.drop("MSFT", axis=0)
            if "MSFT" in data.index:
                self.msft_ma5.append(data.loc["MSFT", "ma5"])
            weights = data.ma5 / data.ma5.sum()
            self.blotter.batch_order_target_percent(data.index, weights)

    results = trading.run_backtest(adjusted_daily_loader, OneEngineAlg, "2019-01-11", "2019-01-15")
    return results, OneEngineAlg.msft_ma5


def test_one_engine_algorithm(one_engine_results):
    results, msft_ma5 = one_engine_results
    # MA5 of 01-08 to 01-14, adjusted by the 1-for-15 reverse split on 01-14
    assert msft_ma5 == [pytest.approx(1268.4657576628665)]

    # --- day1, factors are delayed, only AAPL ---
    aapl_shares1 = int(round(1e5 / 155.19))
    aapl_cost1 = aapl_shares1 * 155.19
    cash1 = 1e5 - aapl_cost1
    aapl_value_eod1 = aapl_shares1 * 157
    # --- day2 ---
    aapl_weight2 = 155.854 / (155.854 + 1268.466)
    msft_weight2 = 1268.466 / (155.854 + 1268.466)
    value_bod2 = aapl_shares1 * 150.81 + cash1
    aapl_shares_change = int(round(aapl_weight2 * value_bod2 / 150.81)) - aapl_shares1
    aapl_shares2 = aapl_shares1 + aapl_shares_change
    aapl_basis = (155.19 * aapl_shares1 + aapl_shares_change * 150.81) / aapl_shares2
    aapl_value2 = aapl_shares2 * 156.94
    msft_shares2 = int(round(msft_weight2 * value_bod2 / 103.19))
    msft_value2 = msft_shares2 * 108.85
    cash2 = 1e5 - aapl_cost1 + (aapl_shares1 - aapl_shares2) * 150.81 - msft_shares2 * 103.19
    expected = pd.DataFrame(
        [
            [nan, nan, nan, nan, nan, nan, 100000.00],
            [155.19, nan, aapl_shares1, nan, aapl_value_eod1, nan, cash1],
            [aapl_basis, 103.19, aapl_shares2, msft_shares2, aapl_value2, msft_value2, cash2],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("avg_px", "AAPL"),
                ("avg_px", "MSFT"),
                ("shares", "AAPL"),
                ("shares", "MSFT"),
                ("value", "AAPL"),
                ("value", "MSFT"),
                ("value", "cash"),
            ]
        ),
        index=[ts("2019-01-13"), ts("2019-01-14"), ts("2019-01-15")],
    )
    expected.index.name = "index"
    pd.testing.assert_frame_equal(expected, results.positions)

    # returns
    value = expected.value.sum(axis=1)
    assert_almost_equal(results.returns.values, (value / value.shift(1) - 1).values)
    assert list(results.transactions.symbol) == ["AAPL", "AAPL", "MSFT"]


def test_two_engine_algorithm(adjusted_daily_loader):
    class TwoEngineAlg(trading.CustomAlgorithm):
        def initialize(self):
            engine_main = self.get_factor_engine("main")
            engine_test = self.get_factor_engine("test")
            engine_main.add(factors.MA(5), "ma5")
            engine_test.add(factors.MA(4), "ma4")

            self.schedule_rebalance(trading.event.MarketClose(self.rebalance, offset_ns=-10000))
            self.blotter.set_commission(0, 0.005, 1)
            self.blotter.set_slippage(0, 0.4)

        def rebalance(self, data, history):
            mask = data["test"].ma4 > data["main"].ma5
            masked_test = data["test"][mask]
            weights = masked_test.ma4 / masked_test.ma4.sum()
            for asset, weight in zip(masked_test.index, weights):
                self.blotter.order_target_percent(asset, weight)

    blotter = trading.SimulationBlotter(adjusted_daily_loader)
    evt_mgr = trading.SimulationEventManager()
    alg = TwoEngineAlg(blotter, main=adjusted_daily_loader, test=adjusted_daily_loader)
    evt_mgr.subscribe(alg)
    evt_mgr.subscribe(blotter)

    evt_mgr.run("2019-01-10", "2019-01-15")
    first_run = str(blotter)
    evt_mgr.run("2019-01-10", "2019-01-15")
    # run twice should get the same result
    assert str(blotter) == first_run
    assert_array_equal(
        blotter.get_transactions().symbol.values, ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL"]
    )


def test_record():
    recorder = trading.CustomAlgorithm(None, main=None)._recorder
    recorder.record("2019-01-10", dict(a=1, b=2))
    recorder.record("2019-01-11", dict(a=2, b=3, c=4))
    expected = pd.DataFrame(
        [[1, 2, nan], [2, 3, 4]], columns=["a", "b", "c"], index=["2019-01-10", "2019-01-11"]
    )
    expected.index.name = "date"
    pd.testing.assert_frame_equal(expected, recorder.to_df())

    with pytest.raises(ValueError, match="reserved key"):
        recorder.record("2019-01-12", dict(date=1))


def test_algorithm_needs_data_source():
    with pytest.raises(ValueError, match="At least one data source"):
        trading.CustomAlgorithm(None)


@pytest.mark.parametrize(
    "event, expected_prices",
    [
        # rebalance 10000 ns before market close (the close of the last bar of the day)
        ("close", [157.92, 142.09, 148.26]),
        # rebalance at market open (the close of the first bar of the day)
        ("open", [143.95, 144.58]),
    ],
)
def test_intraday_algorithm(intraday_loader, event, expected_prices):
    class IntradayAlg(trading.CustomAlgorithm):
        order_shares = 0.3

        def initialize(self):
            self.get_factor_engine().add(factors.MA(5), "ma5")
            if event == "close":
                self.schedule_rebalance(trading.event.MarketClose(self.rebalance, offset_ns=-10000))
            else:
                self.schedule_rebalance(trading.event.MarketOpen(self.rebalance))
            self.blotter.set_commission(0, 0.005, 1)
            self.blotter.set_slippage(0, 0.4)

        def rebalance(self, data, history):
            self.blotter.order_target_percent("AAPL", self.order_shares)
            self.order_shares = -self.order_shares

    results = trading.run_backtest(intraday_loader, IntradayAlg, "2019-01-01", "2019-01-05")
    assert_almost_equal(results.transactions.price.values, expected_prices)
    if event == "close":
        assert list(results.transactions.index) == [
            ts("2019-01-02 20:55:00"),
            ts("2019-01-03 20:55:00"),
            ts("2019-01-04 20:55:00"),
        ]


# ------------------------------- get_price_matrix inside a backtest -------------------------------

OHLCV = ("open", "high", "low", "close", "volume")
BEFORE_OPEN = {"offset_ns": -1}
BEFORE_CLOSE = {"offset_ns": -10000}


def make_panel(dates, assets="ABC", seed=0):
    """A random walk, with different open, high, low, close and volume on every bar."""
    rng = np.random.default_rng(seed)
    shape = len(dates), len(assets)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0))
    close = pd.DataFrame(close, index=dates, columns=list(assets))
    opens = close * np.exp(rng.normal(0, 0.01, shape))
    columns = {
        "open": opens,
        "high": np.maximum(opens, close) * 1.01,
        "low": np.minimum(opens, close) * 0.99,
        "close": close,
        "volume": close * rng.uniform(1e3, 1e4, shape),
    }
    panel = pd.concat({col: df.stack() for col, df in columns.items()}, axis=1)
    return panel.rename_axis(["date", "asset"])


def make_loader(panel, **kwargs):
    return spectre.data.MemoryLoader(panel, ohlcv=OHLCV, **kwargs)


def matrices_at(event, loader, start, end, length, prices, universe=None, device="cpu"):
    """What `get_price_matrix` returns inside `event`, by the time of the call."""
    seen = {}

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            engine = self.get_factor_engine()
            if device == "cuda":
                engine.to_cuda()
            engine.add(factors.OHLCV.open, "open")
            if universe is not None:
                engine.set_filter(universe)
            self.schedule_rebalance(event(self.rebalance))

        def rebalance(self, data, history):
            seen[self.current] = self.get_price_matrix(length, prices=prices), data

    trading.run_backtest(loader, Alg, start, end)
    return seen


def test_future_close_does_not_change_a_rebalance_at_the_open(device):
    dates = pd.bdate_range("2020-01-01", periods=30, tz="UTC")
    start, bar = dates[10], 20
    panel = make_panel(dates)
    changed = panel.copy()
    changed.loc[(dates[bar], "A"), "close"] *= 1.5

    def run(prices):
        seen = {}

        class Alg(trading.CustomAlgorithm):
            def initialize(self):
                engine = self.get_factor_engine()
                if device == "cuda":
                    engine.to_cuda()
                engine.add(factors.OHLCV.open, "open")
                self.blotter.set_commission(0, 0, 0)
                self.blotter.set_slippage(0, 0)
                self.schedule_rebalance(trading.event.MarketOpen(self.rebalance))

            def rebalance(self, data, history):
                matrix = self.get_price_matrix(pd.DateOffset(days=7))
                seen[self.current] = matrix
                # hold the asset that the last close in the matrix puts the highest over the open
                best = (matrix.iloc[-1] / data.open).idxmax()
                weights = (data.index == best).astype(float)
                self.blotter.batch_order_target_percent(data.index, weights)

        return trading.run_backtest(make_loader(prices), Alg, start, dates[-1]), seen

    results, seen = run(panel)
    other_results, other_seen = run(changed)

    def orders_up_to_the_bar(run_results):
        orders = run_results.transactions[: dates[bar]].reset_index()
        return orders.sort_values(["index", "symbol"]).reset_index(drop=True)

    for date in dates[10 : bar + 1]:
        pd.testing.assert_frame_equal(seen[date], other_seen[date])
    pd.testing.assert_frame_equal(
        orders_up_to_the_bar(results), orders_up_to_the_bar(other_results)
    )
    # the next open knows it, so the check above was able to fail
    assert not seen[dates[bar + 1]].equals(other_seen[dates[bar + 1]])


@pytest.mark.parametrize(
    "event, kwargs",
    [
        # fires before the open on daily bars
        (trading.event.EveryBarData, {}),
        (trading.event.MarketOpen, BEFORE_OPEN),
        (trading.event.MarketOpen, {}),
    ],
    ids=["every_bar", "before_open", "open"],
)
@pytest.mark.parametrize("column", ["high", "low", "close", "volume"])
def test_price_matrix_stops_before_the_bar_that_has_not_closed(event, kwargs, column):
    dates = pd.bdate_range("2020-01-01", periods=16, tz="UTC")
    panel = make_panel(dates)
    prices = getattr(factors.OHLCV, column)
    length = pd.DateOffset(days=6)

    seen = matrices_at(
        lambda cb: event(cb, **kwargs), make_loader(panel), dates[8], dates[-1], length, prices
    )

    assert list(seen) == list(dates[8:])
    table = panel[column].unstack()
    for date, (matrix, _) in seen.items():
        expected = table.loc[date - length : date].iloc[:-1]
        assert list(matrix.index) == list(expected.index)
        assert_almost_equal(matrix.values, expected.values, decimal=3)


@pytest.mark.parametrize(
    "event, kwargs",
    [
        (trading.event.EveryBarData, {}),
        (trading.event.MarketOpen, BEFORE_OPEN),
        (trading.event.MarketOpen, {}),
    ],
    ids=["every_bar", "before_open", "open"],
)
def test_price_matrix_has_the_open_of_the_current_bar(event, kwargs):
    dates = pd.bdate_range("2020-01-01", periods=16, tz="UTC")
    panel = make_panel(dates)
    length = pd.DateOffset(days=6)

    seen = matrices_at(
        lambda cb: event(cb, **kwargs),
        make_loader(panel),
        dates[8],
        dates[-1],
        length,
        factors.OHLCV.open,
    )

    assert list(seen) == list(dates[8:])
    table = panel["open"].unstack()
    for date, (matrix, _) in seen.items():
        expected = table.loc[date - length : date]
        assert matrix.index[-1] == date
        assert_almost_equal(matrix.values, expected.values, decimal=3)


@pytest.mark.parametrize("kwargs", [BEFORE_CLOSE, {}], ids=["before_close", "after_close"])
def test_price_matrix_has_the_close_at_the_market_close(kwargs):
    dates = pd.bdate_range("2020-01-01", periods=16, tz="UTC")
    panel = make_panel(dates)
    length = pd.DateOffset(days=6)

    seen = matrices_at(
        lambda cb: trading.event.MarketClose(cb, **kwargs),
        make_loader(panel),
        dates[8],
        dates[-1],
        length,
        factors.OHLCV.close,
    )

    assert list(seen) == list(dates[8:])
    table = panel["close"].unstack()
    for date, (matrix, _) in seen.items():
        assert_almost_equal(matrix.values, table.loc[date - length : date].values, decimal=3)


def test_price_matrix_before_the_close_shows_the_price_of_the_blotter():
    """Before the close the blotter has moved to the close already, and it fills there."""
    dates = pd.bdate_range("2020-01-01", periods=16, tz="UTC")
    last_rows = {}

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            self.get_factor_engine().add(factors.OHLCV.open, "open")
            self.schedule_rebalance(trading.event.MarketClose(self.rebalance, **BEFORE_CLOSE))

        def rebalance(self, data, history):
            matrix = self.get_price_matrix(pd.DateOffset(days=6))
            last_rows[self.current] = matrix["A"].iloc[-1], self.blotter.get_price("A")

    trading.run_backtest(make_loader(make_panel(dates)), Alg, dates[8], dates[-1])

    assert list(last_rows) == list(dates[8:])
    for in_matrix, in_blotter in last_rows.values():
        assert in_matrix == pytest.approx(in_blotter, rel=1e-6)


def test_price_matrix_on_intraday_bars():
    """An intraday bar is complete when its EveryBarData fires, the blotter is at its close."""
    days = pd.bdate_range("2020-01-06", periods=4, tz="UTC")
    dates = pd.DatetimeIndex(
        [day + pd.Timedelta(minutes=870 + 5 * i) for day in days for i in range(3)]
    )
    panel = make_panel(dates)
    bar_before = dict(zip(dates[1:], dates[:-1]))
    length, start = pd.DateOffset(days=1), days[2]
    loader = make_loader(panel)

    def last_rows(event, prices=factors.OHLCV.close):
        seen = matrices_at(event, loader, start, dates[-1], length, prices)
        return {date: matrix.index[-1] for date, (matrix, _) in seen.items()}

    # the first open of a run has no data yet, it comes with the first EveryBarData
    open_bar = days[3] + pd.Timedelta(minutes=870)
    assert last_rows(trading.event.MarketOpen) == {open_bar: bar_before[open_bar]}
    assert last_rows(trading.event.MarketOpen, factors.OHLCV.open) == {open_bar: open_bar}

    bars = dates[dates >= start]
    assert last_rows(trading.event.EveryBarData) == {date: date for date in bars}

    last_bars = [day + pd.Timedelta(minutes=880) for day in days[2:]]
    close_rows = last_rows(lambda cb: trading.event.MarketClose(cb, **BEFORE_CLOSE))
    assert close_rows == {date: date for date in last_bars}


@pytest.mark.parametrize(
    "prices, has_current_bar",
    [(factors.OHLCV.close, False), (factors.OHLCV.open, True)],
    ids=["close", "open"],
)
def test_filter_of_price_matrix_does_not_read_a_future_close(device, prices, has_current_bar):
    dates = pd.bdate_range("2020-01-01", periods=14, tz="UTC")
    panel = make_panel(dates, "AB")
    panel["close"] = panel["open"] = 100.0
    # B reaches the universe on the close of `bar` and stays there, A is in from the start
    bar = 9
    panel.loc[(dates, "A"), "close"] = 300.0
    panel.loc[(dates[bar:], "B"), "close"] = 300.0
    start = dates[6]

    seen = matrices_at(
        trading.event.MarketOpen,
        make_loader(panel),
        start,
        dates[-1],
        pd.DateOffset(days=3),
        prices,
        universe=factors.OHLCV.close >= 200,
        device=device,
    )

    assert list(seen) == list(dates[7:])
    for date, (matrix, data) in seen.items():
        assert list(matrix.columns) == (["A", "B"] if date > dates[bar] else ["A"])
        # what the rebalance is told about is in the matrix
        assert set(data.index) <= set(matrix.columns)
        assert (matrix.index[-1] == date) == has_current_bar


def test_filter_on_the_open_keeps_the_current_bar_in_price_matrix(device):
    dates = pd.bdate_range("2020-01-01", periods=14, tz="UTC")
    panel = make_panel(dates, "AB")
    panel["open"] = 100.0
    # the open of the current bar is known at the open, so the universe has it
    bar = 9
    panel.loc[(dates, "A"), "open"] = 300.0
    panel.loc[(dates[bar:], "B"), "open"] = 300.0

    seen = matrices_at(
        trading.event.MarketOpen,
        make_loader(panel),
        dates[6],
        dates[-1],
        pd.DateOffset(days=3),
        factors.OHLCV.close,
        universe=factors.OHLCV.open >= 200,
        device=device,
    )

    for date, (matrix, data) in seen.items():
        assert list(data.index) == (["A", "B"] if date >= dates[bar] else ["A"])
        assert list(matrix.columns) == list(data.index)


def test_price_matrix_at_the_open_is_adjusted_to_the_current_bar(device):
    dates = pd.bdate_range("2020-01-01", periods=14, tz="UTC")
    panel = make_panel(dates, "AB")
    bar = 9
    prices = ["open", "high", "low", "close"]
    panel.loc[(dates[bar:], "A"), prices] *= 0.5
    panel = panel.assign(**{"ex-dividend": 0.0, "split_ratio": 1.0})
    panel.loc[(dates[bar], "A"), "split_ratio"] = 0.5
    loader = make_loader(panel, adjustments=("ex-dividend", "split_ratio"))
    length = pd.DateOffset(days=3)

    seen = matrices_at(
        trading.event.MarketOpen,
        loader,
        dates[6],
        dates[-1],
        length,
        factors.OHLCV.close,
        device=device,
    )

    def closes_before(date):
        return panel["close"].unstack().loc[date - length : date].iloc[:-1]

    # the bar before the ex-date doesn't know the split yet
    date = dates[bar - 1]
    assert_almost_equal(seen[date][0].values, closes_before(date).values, 3)
    # the open of the ex-date does, the blotter trades the new price
    date = dates[bar]
    expected = closes_before(date)
    assert_almost_equal(seen[date][0].A.values, expected.A.values * 0.5, 3)
    assert_almost_equal(seen[date][0].B.values, expected.B.values, 3)


def test_price_matrix_gives_the_filter_back_after_an_error():
    dates = pd.bdate_range("2020-01-01", periods=14, tz="UTC")
    loader = make_loader(make_panel(dates))
    universe = factors.OHLCV.close >= 0

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            engine = self.get_factor_engine()
            engine.add(factors.OHLCV.open, "open")
            engine.set_filter(universe)
            self.schedule_rebalance(trading.event.MarketOpen(self.rebalance))

        def rebalance(self, data, history):
            self.get_price_matrix(pd.DateOffset(days=3))
            assert self.get_factor_engine().get_filter() is universe
            self.get_price_matrix(
                pd.DateOffset(days=3), prices=factors.ColumnDataFactor(inputs=("no such column",))
            )

    blotter = trading.SimulationBlotter(loader)
    alg = Alg(blotter, main=loader)
    evt_mgr = trading.SimulationEventManager()
    evt_mgr.subscribe(blotter)
    evt_mgr.subscribe(alg)
    with pytest.raises(KeyError):
        evt_mgr.run(dates[6], dates[-1])
    assert alg.get_factor_engine().get_filter() is universe


def test_price_matrix_leaves_a_filter_that_reads_no_close_alone(capsys):
    """StaticAssets doesn't read a close, and setting it again empties the engine cache."""
    dates = pd.bdate_range("2020-01-01", periods=14, tz="UTC")

    seen = matrices_at(
        trading.event.MarketOpen,
        make_loader(make_panel(dates)),
        dates[6],
        dates[-1],
        pd.DateOffset(days=3),
        factors.OHLCV.close,
        universe=factors.StaticAssets({"A", "B"}),
    )

    assert all(list(matrix.columns) == ["A", "B"] for matrix, _ in seen.values())
    assert "Cache clean" not in capsys.readouterr().out


# ------------------------------------- the simulation loop --------------------------------------


def flat_panel(dates, dividend_on, split_on):
    """A at 100 pays 1 on `dividend_on`, B at 100 splits 2:1 on `split_on`, to 50."""
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    panel = pd.DataFrame(100.0, index=index, columns=list(OHLCV))
    panel = panel.assign(**{"volume": 1e6, "ex-dividend": 0.0, "split_ratio": 1.0})
    panel.loc[(dates[dates >= split_on], "B"), ["open", "high", "low", "close"]] = 50.0
    panel.loc[(split_on, "B"), "split_ratio"] = 0.5
    panel.loc[(dividend_on, "A"), "ex-dividend"] = 1.0
    return panel


def monthly_bars():
    dates = pd.date_range("2019-01-01", periods=12, freq="MS", tz="UTC")
    return flat_panel(dates, ts("2019-06-01"), ts("2019-09-01")), dates, dates, None


def daily_bars_filtered_to_the_15th():
    days = pd.bdate_range("2019-01-01", "2019-12-31", tz="UTC")
    panel = flat_panel(days, ts("2019-05-16"), ts("2019-08-16"))
    panel["flag"] = (panel.index.get_level_values(0).day == 15).astype(float)
    flag = factors.ColumnDataFactor(inputs=("flag",), should_delay=False)
    fifteenths = days[days.day == 15]
    return panel, fifteenths, fifteenths, flag > 0.5


def intraday_bars_a_month_apart():
    days = pd.DatetimeIndex([ts(f"2019-0{month}-15") for month in range(1, 7)])
    first, last = (
        days + pd.Timedelta(hours=14, minutes=30),
        days + pd.Timedelta(hours=14, minutes=35),
    )
    dates = first.append(last).sort_values()
    return flat_panel(dates, first[2], first[4]), first, last, None


@pytest.mark.parametrize(
    "bars", [monthly_bars, daily_bars_filtered_to_the_15th, intraday_bars_a_month_apart]
)
def test_ticks_a_month_apart_are_different_days(bars):
    panel, first_bars, last_bars, universe = bars()
    adjustments = ("ex-dividend", "split_ratio")
    opens, closes = [], []

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            engine = self.get_factor_engine()
            engine.add(factors.OHLCV.open, "open")
            if universe is not None:
                engine.set_filter(universe)
            self.blotter.set_commission(0, 0, 0)
            self.blotter.set_slippage(0, 0)
            self.blotter.borrow_money_interest_rate = 0
            self.blotter.borrow_stock_interest_rate = 0
            self.schedule(trading.event.MarketOpen(self.on_open))
            self.schedule(trading.event.MarketClose(self.on_close))

        def on_open(self, _):
            opens.append(self.current)
            if not self.blotter.positions:
                self.blotter.order("A", 100)
                self.blotter.order("B", 100)

        def on_close(self, _):
            closes.append(self.current)

    loader = make_loader(panel, adjustments=adjustments)
    results = trading.run_backtest(loader, Alg, first_bars[0], last_bars[-1])

    assert opens == list(first_bars)
    assert closes == list(last_bars)
    last = results.positions.iloc[-1]
    assert last[("shares", "A")] == 100
    assert last[("shares", "B")] == 200
    assert last[("value", "cash")] == pytest.approx(100000 - 20000 + 100)
    assert last[("value", "B")] == pytest.approx(10000)


# --------------------------------- running the same objects again ---------------------------------


class RerunAlg(trading.CustomAlgorithm):
    """Trades on a factor and logs each rebalance. Calls `trip` once, on its 5th rebalance."""

    trip = None
    records = None

    def initialize(self):
        self.seen = []
        self.get_factor_engine().add(factors.MA(5), "ma")
        self.blotter.set_commission(0, 0, 0)
        self.blotter.set_slippage(0, 0)
        self.schedule_rebalance(trading.event.MarketClose(self.rebalance, **BEFORE_CLOSE))

    def rebalance(self, data, history):
        self.seen.append(self.current)
        self.record(ma=float(data.ma.mean()))
        if self.trip is not None and len(self.seen) == 5:
            trip, self.trip = self.trip, None
            trip(self)
        self.blotter.batch_order_target_percent(data.index, data.ma / data.ma.sum())

    def terminate(self, records):
        self.records = records


def new_rerun_alg(loader, alg_type=RerunAlg):
    blotter = trading.SimulationBlotter(loader)
    alg = alg_type(blotter, main=loader)
    evt_mgr = trading.SimulationEventManager()
    evt_mgr.subscribe(blotter)
    evt_mgr.subscribe(alg)
    return alg, evt_mgr


@pytest.fixture
def rerun():
    """A loader, the range of its run, and a reference run of that range on new objects."""
    dates = pd.bdate_range("2020-01-01", periods=40, tz="UTC")
    loader = make_loader(make_panel(dates))
    start, end = dates[10], dates[-1]
    reference, evt_mgr = new_rerun_alg(loader)
    evt_mgr.run(start, end)
    return loader, start, end, reference


def assert_same_run(alg, reference):
    assert alg.seen == reference.seen
    pd.testing.assert_series_equal(alg.results.returns, reference.results.returns)
    pd.testing.assert_frame_equal(alg.results.positions, reference.results.positions)
    pd.testing.assert_frame_equal(alg.results.transactions, reference.results.transactions)


def raise_keyboard_interrupt(_):
    raise KeyboardInterrupt


def raise_value_error(_):
    raise ValueError("a bug in rebalance")


@pytest.mark.parametrize(
    "trip, raises",
    [(raise_keyboard_interrupt, None), (raise_value_error, ValueError)],
    ids=["interrupted", "exception"],
)
def test_run_after_a_broken_run_is_complete(rerun, trip, raises):
    loader, start, end, reference = rerun
    assert len(reference.seen) == 29

    alg, evt_mgr = new_rerun_alg(loader)
    alg.trip = trip
    if raises is None:
        evt_mgr.run(start, end)
    else:
        with pytest.raises(raises):
            evt_mgr.run(start, end)
    assert len(alg.seen) == 5

    evt_mgr.run(start, end)
    assert_same_run(alg, reference)


def test_records_are_of_the_last_run(rerun):
    loader, start, end, reference = rerun
    alg, evt_mgr = new_rerun_alg(loader)
    evt_mgr.run(start, end)
    pd.testing.assert_frame_equal(alg.records, reference.records)

    shorter_end = reference.seen[9]
    evt_mgr.run(start, shorter_end)

    assert list(alg.records.index) == reference.seen[:10]
    pd.testing.assert_frame_equal(alg.records, reference.records[:shorter_end])
    # they are still there after the run
    pd.testing.assert_frame_equal(alg._recorder.to_df(), alg.records)


def test_records_of_an_interrupted_run_reach_terminate(rerun):
    loader, start, end, reference = rerun
    alg, evt_mgr = new_rerun_alg(loader)
    evt_mgr.run(start, end)

    alg.trip = raise_keyboard_interrupt
    evt_mgr.run(start, end)

    assert list(alg.records.index) == reference.seen[:5]


def test_run_after_a_stop_is_complete(rerun):
    loader, start, end, reference = rerun
    alg, evt_mgr = new_rerun_alg(loader)
    alg.trip = lambda algorithm: algorithm.stop_event_manager()
    evt_mgr.run(start, end)
    # it ends the bar it stopped on, so one more close
    assert alg.seen == reference.seen[:6]

    evt_mgr.run(start, end)
    assert_same_run(alg, reference)


class SecondEngineAlg(trading.CustomAlgorithm):
    """Makes an engine of its own in `initialize`, which runs on every run."""

    fail = False

    def initialize(self):
        self.seen = []
        self.get_factor_engine("main").add(factors.MA(5), "ma")
        self.create_factor_engine("second").add(factors.MA(3), "ma")
        if self.fail:
            self.fail = False
            raise ValueError("a bug in initialize")
        self.blotter.set_commission(0, 0, 0)
        self.blotter.set_slippage(0, 0)
        self.schedule_rebalance(trading.event.MarketClose(self.rebalance, **BEFORE_CLOSE))

    def rebalance(self, data, history):
        self.seen.append(self.current)
        weights = data["main"].ma * data["second"].ma
        self.blotter.batch_order_target_percent(weights.index, weights / weights.sum())


def test_engine_created_in_initialize_works_on_a_second_run(rerun):
    loader, start, end, _ = rerun
    fresh, evt_mgr = new_rerun_alg(loader, SecondEngineAlg)
    evt_mgr.run(start, end)
    assert len(fresh.seen) == 29

    alg, evt_mgr = new_rerun_alg(loader, SecondEngineAlg)
    evt_mgr.run(start, end)
    first_engine = alg.get_factor_engine("second")
    evt_mgr.run(start, end)

    assert_same_run(alg, fresh)
    assert alg.get_factor_engine("second") is not first_engine


def test_engine_created_in_initialize_after_a_failed_initialize(rerun):
    loader, start, end, reference = rerun
    alg, evt_mgr = new_rerun_alg(loader, SecondEngineAlg)
    alg.fail = True
    with pytest.raises(ValueError, match="a bug in initialize"):
        evt_mgr.run(start, end)

    evt_mgr.run(start, end)

    assert len(alg.seen) == len(reference.seen)


def test_engine_name_used_twice_in_a_run_is_still_an_error(rerun):
    class TwiceAlg(SecondEngineAlg):
        def initialize(self):
            super().initialize()
            self.create_factor_engine("second")

    loader, start, end, _ = rerun
    _, evt_mgr = new_rerun_alg(loader, TwiceAlg)
    with pytest.raises(AssertionError):
        evt_mgr.run(start, end)


def test_engine_name_of_a_data_source_is_still_an_error(rerun):
    class SourceNameAlg(SecondEngineAlg):
        def initialize(self):
            self.create_factor_engine("main")

    loader, start, end, _ = rerun
    for _ in range(2):
        _, evt_mgr = new_rerun_alg(loader, SourceNameAlg)
        with pytest.raises(AssertionError):
            evt_mgr.run(start, end)


def test_engine_created_before_a_run_stays(rerun):
    class UsesItsEngineAlg(RerunAlg):
        def initialize(self):
            super().initialize()
            self.get_factor_engine("mine").add(factors.MA(3), "ma")

        def rebalance(self, data, history):
            super().rebalance(data["main"], history)

    loader, start, end, _ = rerun
    alg, evt_mgr = new_rerun_alg(loader, UsesItsEngineAlg)
    engine = alg.create_factor_engine("mine")

    evt_mgr.run(start, end)
    evt_mgr.run(start, end)

    assert alg.get_factor_engine("mine") is engine
    assert len(alg.seen) == 29


class InterruptedAlg(RerunAlg):
    """Gets Ctrl-C while the engine computes its factors, before the first bar of a run."""

    interrupt = False
    terminated = False

    def initialize(self):
        super().initialize()
        alg = self

        class Interrupted(factors.CustomFactor):
            def compute(self, data):
                if alg.interrupt:
                    alg.interrupt = False
                    raise KeyboardInterrupt
                return data

        engine = self.get_factor_engine()
        engine.add(Interrupted(inputs=[factors.OHLCV.close]), "interrupted")

    def terminate(self, records):
        super().terminate(records)
        self.terminated = True


def test_interrupt_while_the_factors_are_computed_is_a_clean_stop(rerun, capsys):
    loader, start, end, _ = rerun
    alg, evt_mgr = new_rerun_alg(loader, InterruptedAlg)
    alg.interrupt = True

    evt_mgr.run(start, end)

    assert "Interrupted" in capsys.readouterr().out
    assert alg.seen == []
    assert alg.results == (None, None, None)
    # it would read the results, and there are none
    assert not alg.terminated


def test_run_after_an_interrupt_while_the_factors_are_computed(rerun):
    loader, start, end, _ = rerun
    fresh, evt_mgr = new_rerun_alg(loader, InterruptedAlg)
    evt_mgr.run(start, end)

    alg, evt_mgr = new_rerun_alg(loader, InterruptedAlg)
    alg.interrupt = True
    evt_mgr.run(start, end)
    evt_mgr.run(start, end)

    assert_same_run(alg, fresh)
    assert alg.terminated


def test_interrupted_run_does_not_keep_the_results_of_the_last_run(rerun):
    loader, start, end, _ = rerun
    alg, evt_mgr = new_rerun_alg(loader, InterruptedAlg)
    evt_mgr.run(start, end)
    assert alg.results.returns is not None

    alg.interrupt = True
    evt_mgr.run(start, end)

    assert alg.results == (None, None, None)


class OpenAlg(trading.CustomAlgorithm):
    """Trades at each open, and logs the time of the last row each rebalance was given."""

    def initialize(self):
        self.seen = []
        self.last_rows = []
        self.get_factor_engine().add(factors.OHLCV.open, "open")
        self.blotter.set_commission(0, 0, 0)
        self.blotter.set_slippage(0, 0)
        self.schedule_rebalance(trading.event.MarketOpen(self.rebalance))

    def rebalance(self, data, history):
        self.seen.append(self.current)
        self.last_rows.append(history.index.get_level_values(0)[-1])
        self.blotter.batch_order_target_percent(data.index, data.open / data.open.sum())


@pytest.mark.parametrize("first_bar", [0, 5], ids=["first_bar_of_the_data", "later_bar"])
def test_plot_with_an_asset_as_the_benchmark(monkeypatch, first_bar):
    dates = pd.bdate_range("2020-01-01", periods=30, tz="UTC")
    panel = make_panel(dates)
    alg, evt_mgr = new_rerun_alg(make_loader(panel), OpenAlg)
    evt_mgr.run(dates[first_bar], dates[-1])
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: shown.append(self))

    alg.plot(benchmark="A")

    line = next(trace for trace in shown[0].data if trace.name == "benchmark")
    # returns of the benchmark from the first bar the line has, so its first point has none
    line_dates = pd.DatetimeIndex(line.x)
    closes = panel["close"].unstack()["A"].loc[line_dates[0] :]
    expected = (closes / closes.iloc[0] - 1).iloc[1:].to_numpy() * 100
    assert np.asarray(line.y[1:], dtype=float) == pytest.approx(expected, rel=1e-3)
    assert line_dates[-1] == dates[-1]
    assert line_dates[0] <= dates[first_bar]


def test_rerun_on_intraday_bars_does_not_see_the_end_of_the_last_run():
    days = pd.bdate_range("2020-01-06", periods=5, tz="UTC")
    dates = pd.DatetimeIndex(
        [day + pd.Timedelta(minutes=870 + 5 * i) for day in days for i in range(3)]
    )
    loader = make_loader(make_panel(dates))
    start, end = days[1], dates[-1]
    fresh, evt_mgr = new_rerun_alg(loader, OpenAlg)
    evt_mgr.run(start, end)

    alg, evt_mgr = new_rerun_alg(loader, OpenAlg)
    evt_mgr.run(start, end)
    evt_mgr.run(start, end)

    # the bars of the first open come with its first EveryBarData, after it
    assert fresh.seen == list(days[2:] + pd.Timedelta(minutes=870))
    assert all(last_row < now for now, last_row in zip(alg.seen, alg.last_rows))
    assert alg.last_rows == fresh.last_rows
    assert_same_run(alg, fresh)


def rows_run_engine_shows_at_the_close(loader, start, end, **kwargs):
    last_rows = {}

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            self.get_factor_engine().add(factors.OHLCV.open, "open")
            self.schedule_rebalance(trading.event.MarketClose(self.rebalance, **kwargs))

        def rebalance(self, data, history):
            frame, last = self.run_engine(None, None)
            last_rows[self.current] = frame.index.get_level_values(0)[-1]
            pd.testing.assert_frame_equal(last, frame.loc[frame.index[-1][0]])

    trading.run_backtest(loader, Alg, start, end)
    return last_rows


@pytest.mark.parametrize("kwargs", [BEFORE_CLOSE, {}], ids=["before_close", "after_close"])
def test_run_engine_inside_market_close_ends_at_the_closing_bar(kwargs):
    dates = pd.bdate_range("2020-01-01", periods=12, tz="UTC")
    loader = make_loader(make_panel(dates))

    last_rows = rows_run_engine_shows_at_the_close(loader, dates[3], dates[-1], **kwargs)

    assert last_rows == {date: date for date in dates[3:]}


@pytest.mark.parametrize("kwargs", [BEFORE_CLOSE, {}], ids=["before_close", "after_close"])
def test_run_engine_inside_market_close_ends_at_the_closing_bar_of_the_day(kwargs):
    days = pd.bdate_range("2020-01-06", periods=4, tz="UTC")
    dates = pd.DatetimeIndex(
        [day + pd.Timedelta(minutes=870 + 5 * i) for day in days for i in range(3)]
    )
    loader = make_loader(make_panel(dates))

    last_rows = rows_run_engine_shows_at_the_close(loader, days[1], dates[-1], **kwargs)

    closes = days[1:] + pd.Timedelta(minutes=880)
    assert last_rows == {date: date for date in closes}


# ------------------------------ a second data source of longer bars ------------------------------


def sessions(days):
    """Four bars of 30 minutes on each of `days`."""
    return pd.DatetimeIndex(
        [day + pd.Timedelta(minutes=870 + 30 * i) for day in days for i in range(4)]
    )


def longer_bars(intraday, stamps):
    """The bars of `intraday` added up, by `stamps`, the stamp of the longer bar of each date."""
    dates = intraday.index.get_level_values(0)
    bars = intraday.groupby([stamps(dates), intraday.index.get_level_values(1)])
    bars = bars.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return bars.rename_axis(["date", "asset"])


def daily_stamps(dates):
    return dates.normalize()


def weekly_stamps(dates):
    return dates.normalize() - pd.to_timedelta(dates.day_of_week, unit="D")


def monthly_stamps(dates):
    return pd.DatetimeIndex(dates.tz_localize(None).to_period("M").start_time).tz_localize("UTC")


def two_sources(alg_type, intraday, longer):
    main, second = make_loader(intraday), make_loader(longer)
    blotter = trading.SimulationBlotter(main)
    alg = alg_type(blotter, main=main, longer=second)
    evt_mgr = trading.SimulationEventManager()
    evt_mgr.subscribe(blotter)
    evt_mgr.subscribe(alg)
    return alg, evt_mgr


def longer_matrices_at(
    event, intraday, longer, start, end, prices=None, universe=None, length=None
):
    """What `get_price_matrix` of the second source returns inside `event`, by the time of it."""
    seen = {}
    prices = factors.OHLCV.close if prices is None else prices
    length = pd.DateOffset(days=4) if length is None else length

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            self.get_factor_engine("main").add(factors.OHLCV.open, "open")
            engine = self.get_factor_engine("longer")
            engine.add(factors.OHLCV.open, "open")
            if universe is not None:
                engine.set_filter(universe)
            self.schedule(event(self.look))

        def look(self, _):
            seen[self.current] = self.get_price_matrix(length, name="longer", prices=prices)

    _, evt_mgr = two_sources(Alg, intraday, longer)
    evt_mgr.run(start, end)
    return seen


def last_rows_of(seen):
    return {time: matrix.index[-1] for time, matrix in seen.items()}


def before_close(callback):
    return trading.event.MarketClose(callback, **BEFORE_CLOSE)


def test_later_data_of_the_day_does_not_change_a_rebalance_on_a_second_source(device):
    days = pd.bdate_range("2021-01-04", periods=14, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)
    bar = 10
    first_bar = days[bar] + pd.Timedelta(minutes=870)
    columns = ["high", "low", "close", "volume"]
    changed_daily, changed_intraday = daily.copy(), intraday.copy()
    changed_daily.loc[(days[bar], "A"), columns] *= 1.5
    # all that happens after the first open of the day
    changed_intraday.loc[(first_bar, "A"), columns] *= 1.5
    later = intraday.index.get_level_values(0) > first_bar
    later = later & (intraday.index.get_level_values(1) == "A")
    changed_intraday.loc[later] = changed_intraday.loc[later].to_numpy() * 1.5

    def run(intraday, daily):
        seen = {}

        class Alg(trading.CustomAlgorithm):
            def initialize(self):
                for name in ("main", "longer"):
                    engine = self.get_factor_engine(name)
                    if device == "cuda":
                        engine.to_cuda()
                    engine.add(factors.OHLCV.open, "open")
                self.blotter.set_commission(0, 0, 0)
                self.blotter.set_slippage(0, 0)
                self.schedule(trading.event.MarketOpen(self.look))

            def look(self, _):
                matrix = self.get_price_matrix(pd.DateOffset(days=7), name="longer")
                seen[self.current] = matrix
                assets = list(matrix.columns)
                opens = pd.Series([self.blotter.get_price(asset) for asset in assets], assets)
                # hold the asset that the last close in the matrix puts the highest over the open
                best = (matrix.iloc[-1] / opens).idxmax()
                weights = [float(asset == best) for asset in assets]
                self.blotter.batch_order_target_percent(assets, weights)

        alg, evt_mgr = two_sources(Alg, intraday, daily)
        evt_mgr.run(days[8], days[-1])
        return alg.results, seen

    def orders_up_to_the_open(run_results):
        orders = run_results.transactions[:first_bar].reset_index()
        return orders.sort_values(["index", "symbol"]).reset_index(drop=True)

    results, seen = run(intraday, daily)
    other_results, other_seen = run(changed_intraday, changed_daily)

    assert first_bar in seen
    for time in seen:
        if time <= first_bar:
            pd.testing.assert_frame_equal(seen[time], other_seen[time])
    pd.testing.assert_frame_equal(
        orders_up_to_the_open(results), orders_up_to_the_open(other_results)
    )
    # the next open knows it, so the check above was able to fail
    next_bar = first_bar + pd.Timedelta(days=1)
    assert not seen[next_bar].equals(other_seen[next_bar])


def test_second_source_at_the_open_ends_at_the_day_before():
    days = pd.bdate_range("2021-01-04", periods=9, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)

    seen = longer_matrices_at(trading.event.MarketOpen, intraday, daily, days[4], days[-1])

    opens = days[4:-1] + pd.Timedelta(minutes=870)
    assert last_rows_of(seen) == dict(zip(opens, days[3:-2]))
    closes = daily["close"].unstack()
    for time, matrix in seen.items():
        expected = closes.loc[time - pd.DateOffset(days=4) : time].iloc[:-1]
        assert_almost_equal(matrix.values, expected.values, decimal=3)


def test_second_source_keeps_the_open_of_the_day():
    days = pd.bdate_range("2021-01-04", periods=9, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)

    seen = longer_matrices_at(
        trading.event.MarketOpen, intraday, daily, days[4], days[-1], prices=factors.OHLCV.open
    )

    opens = days[4:-1] + pd.Timedelta(minutes=870)
    assert last_rows_of(seen) == dict(zip(opens, days[4:-1]))


@pytest.mark.parametrize("column", ["high", "low", "volume"])
def test_second_source_hides_all_that_a_close_makes_known(column):
    days = pd.bdate_range("2021-01-04", periods=9, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)

    seen = longer_matrices_at(
        trading.event.MarketOpen,
        intraday,
        daily,
        days[4],
        days[-1],
        prices=getattr(factors.OHLCV, column),
    )

    opens = days[4:-1] + pd.Timedelta(minutes=870)
    assert last_rows_of(seen) == dict(zip(opens, days[3:-2]))


@pytest.mark.parametrize(
    "event", [before_close, trading.event.MarketClose], ids=["before", "after"]
)
def test_second_source_at_the_market_close_has_the_close(event):
    days = pd.bdate_range("2021-01-04", periods=9, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)

    seen = longer_matrices_at(event, intraday, daily, days[4], days[-1])

    closes = days[4:-1] + pd.Timedelta(minutes=960)
    assert last_rows_of(seen) == dict(zip(closes, days[4:-1]))
    for time, matrix in seen.items():
        expected = daily["close"].unstack().loc[time.normalize()]
        assert_almost_equal(matrix.iloc[-1].values, expected.values, decimal=3)


def test_second_source_through_the_day():
    """Each EveryBarData is at the end of its bar, and the last one ends the day."""
    days = pd.bdate_range("2021-01-04", periods=7, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)

    seen = longer_matrices_at(trading.event.EveryBarData, intraday, daily, days[4], days[-1])

    expected = {}
    for day, day_before in zip(days[4:-1], days[3:-2]):
        for bar in sessions([day])[:-1]:
            expected[bar] = day_before
        expected[sessions([day])[-1]] = day
    assert last_rows_of(seen) == expected


def test_second_source_of_weekly_bars_closes_with_the_week():
    days = pd.bdate_range("2021-01-04", periods=25, tz="UTC")
    intraday = make_panel(sessions(days))
    weekly = longer_bars(intraday, weekly_stamps)

    seen = longer_matrices_at(
        before_close, intraday, weekly, days[10], days[20], length=pd.DateOffset(days=14)
    )

    assert len(seen) == 10
    for time, last_row in last_rows_of(seen).items():
        this_week = weekly_stamps(pd.DatetimeIndex([time]))[0]
        friday = time.day_of_week == 4
        assert last_row == (this_week if friday else this_week - pd.Timedelta(days=7))


def test_second_source_of_monthly_bars_closes_with_the_month():
    """A month is longer than the smallest gap between months, so that can't be the bound."""
    days = pd.bdate_range("2021-01-04", "2021-05-03", tz="UTC")
    intraday = make_panel(sessions(days))
    monthly = longer_bars(intraday, monthly_stamps)
    stamps = monthly.index.levels[0]

    seen = longer_matrices_at(
        before_close,
        intraday,
        monthly,
        pd.Timestamp("2021-03-15", tz="UTC"),
        stamps[-1],
        length=pd.DateOffset(days=70),
    )

    assert len(seen) == 35
    for time, last_row in last_rows_of(seen).items():
        started = stamps[stamps <= time]
        is_last_day = time.normalize() == days[days.month == time.month][-1]
        assert last_row == (started[-1] if is_last_day else started[-2])


def test_second_source_has_no_next_bar_for_its_last_row():
    """Its last row lasts for the bar size of the source, not for ever."""
    days = pd.bdate_range("2021-01-04", periods=7, tz="UTC")
    intraday = make_panel(sessions(days))
    first_bars = longer_bars(intraday, lambda dates: dates.normalize() + pd.Timedelta(minutes=870))

    last = sessions(days[-1:])[0]
    seen = longer_matrices_at(trading.event.MarketOpen, intraday, first_bars, days[4], last)

    assert seen[last].index[-1] == days[-2] + pd.Timedelta(minutes=870)


def test_filter_of_a_second_source_does_not_read_a_close_of_today():
    days = pd.bdate_range("2021-01-04", periods=12, tz="UTC")
    intraday = make_panel(sessions(days), "AB")
    daily = longer_bars(intraday, daily_stamps)
    daily["close"] = 100.0
    # B reaches the universe on the close of `bar` and stays there, A is in from the start
    bar = 8
    daily.loc[(days, "A"), "close"] = 300.0
    daily.loc[(days[bar:], "B"), "close"] = 300.0

    seen = longer_matrices_at(
        trading.event.MarketOpen,
        intraday,
        daily,
        days[6],
        days[-1],
        universe=factors.OHLCV.close >= 200,
    )

    for time, matrix in seen.items():
        known = time.normalize() > days[bar]
        assert list(matrix.columns) == (["A", "B"] if known else ["A"])


def test_ticks_removed_by_a_filter_do_not_end_the_day_of_a_second_source():
    """The bars left of the day are still to come, whatever the filter of the ticks does."""
    days = pd.bdate_range("2021-01-04", periods=8, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)
    morning = intraday.index.get_level_values(0).hour == 14
    intraday = intraday.assign(morning=morning.astype(float))
    seen = {}

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            flag = factors.ColumnDataFactor(inputs=("morning",), should_delay=False)
            engine = self.get_factor_engine("main")
            engine.add(factors.OHLCV.open, "open")
            engine.set_filter(flag > 0.5)
            self.get_factor_engine("longer").add(factors.OHLCV.open, "open")
            self.schedule(before_close(self.look))

        def look(self, _):
            seen[self.current] = self.get_price_matrix(pd.DateOffset(days=4), name="longer")

    _, evt_mgr = two_sources(Alg, intraday, daily)
    evt_mgr.run(days[4], days[-1])

    # the last tick of each day is the first bar, and the close of that day is still far
    assert len(seen) == 3
    for time, matrix in seen.items():
        assert time.hour == 14
        assert matrix.index[-1] < time.normalize()


def test_second_source_in_a_timezone():
    days = pd.bdate_range("2021-01-04", periods=9, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)
    seen = {}

    class Alg(trading.CustomAlgorithm):
        def initialize(self):
            for name in ("main", "longer"):
                engine = self.get_factor_engine(name)
                engine.timezone = "America/New_York"
                engine.add(factors.OHLCV.open, "open")
            self.schedule(trading.event.MarketOpen(self.look))

        def look(self, _):
            seen[self.current] = self.get_price_matrix(pd.DateOffset(days=4), name="longer")

    _, evt_mgr = two_sources(Alg, intraday, daily)
    evt_mgr.run(days[4], days[-1])

    assert len(seen) == 4
    for time, matrix in seen.items():
        assert matrix.index[-1] < time.tz_convert("UTC").normalize()


def test_run_engine_and_history_of_a_second_source_are_delayed():
    """They hold today's row at the open, of what was known at the close before."""
    days = pd.bdate_range("2021-01-04", periods=12, tz="UTC")
    intraday = make_panel(sessions(days))
    daily = longer_bars(intraday, daily_stamps)
    bar = 8
    changed = daily.copy()
    changed.loc[(days[bar], "A"), ["high", "low", "close", "volume"]] *= 1.5

    def run(daily):
        seen = {}

        class Alg(trading.CustomAlgorithm):
            def initialize(self):
                self.get_factor_engine("main").add(factors.OHLCV.open, "open")
                self.get_factor_engine("longer").add(factors.MA(2), "ma")
                self.schedule(trading.event.MarketOpen(self.look))

            def look(self, _):
                frames, _ = self.run_engine(None, None)
                seen[self.current] = frames["longer"]

        _, evt_mgr = two_sources(Alg, intraday, daily)
        evt_mgr.run(days[4], days[-1])
        return seen

    seen, other = run(daily), run(changed)

    open_of_the_bar = days[bar] + pd.Timedelta(minutes=870)
    assert seen[open_of_the_bar].index.get_level_values(0)[-1] == days[bar]
    pd.testing.assert_frame_equal(seen[open_of_the_bar], other[open_of_the_bar])
    next_open = days[bar + 1] + pd.Timedelta(minutes=870)
    assert not seen[next_open].equals(other[next_open])
