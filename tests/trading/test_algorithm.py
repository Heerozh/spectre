import pandas as pd
import pytest
from numpy import nan
from numpy.testing import assert_almost_equal, assert_array_equal

from spectre import factors, trading


def ts(date):
    return pd.Timestamp(date, tz="UTC")


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
