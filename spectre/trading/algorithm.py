"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import gc
import os
from abc import ABC
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

from ..data import DataLoader, DataLoaderFastGetter
from ..factors import OHLCV, FactorEngine, NoBarError, StaticAssets
from ..plotting import cumulative_returns_fig
from .blotter import BaseBlotter, SimulationBlotter
from .event import Event, EventManager, EventReceiver, EveryBarData, MarketClose, MarketOpen


class Recorder:
    def __init__(self):
        self._records = []

    def record(self, date, table):
        if "date" in table:
            raise ValueError("`date` is reserved key for record.")
        if np.any([isinstance(v, np.ndarray) for v in table.values()]):
            raise ValueError("recorder does not support np.ndarray datatype")
        table["date"] = date
        self._records.append(table)

    def to_df(self):
        ret = pd.DataFrame(self._records)
        if ret.shape[0] > 0:
            ret = ret.set_index("date").sort_index(axis=0)
        return ret.groupby(level=0).last()


class CustomAlgorithm(EventReceiver, ABC):
    Results = namedtuple("Results", ["returns", "positions", "transactions"])
    """
    Base class for custom trading algorithm.
    """

    def __init__(self, blotter: BaseBlotter, **data_sources: DataLoader):
        """
        :param blotter: order management system for this algorithm.
        :param data_sources: key is data_source_name, value is dataloader
        """
        super().__init__()
        if not data_sources:
            raise ValueError("At least one data source.")

        self._history_window = pd.DateOffset(0)
        self._data = None
        self._last_row = None
        self._engines = {name: FactorEngine(loader) for name, loader in data_sources.items()}
        self._engines_of_initialize = set()
        self.blotter = blotter
        self._recorder = Recorder()
        self._current_dt = None
        # a simulation has the whole of the current bar, this says how much of it has happened.
        # None outside of a simulation, where the data only holds what has happened.
        self._bar_closed = None
        self._bars = {}
        self._results = CustomAlgorithm.Results(None, None, None)
        self._rebalance_callback = None
        self.delay_factor = True

    def clear(self):
        for engine in self._engines.values():
            engine.clear()

    def create_factor_engine(self, name: str, loader: DataLoader = None):
        assert name not in self._engines
        if loader is None:
            def_name = next(iter(self._engines))
            loader = self._engines[def_name].loader_
        self._engines[name] = FactorEngine(loader)
        return self._engines[name]

    def get_factor_engine(self, name: str | None = None):
        if name is None:
            name = next(iter(self._engines))

        if name not in self._engines:
            raise KeyError(
                f"Data source '{name}' not found, please pass in the algorithm "
                f"initialization: `YourAlgorithm({name}=DataLoader())`"
            )
        return self._engines[name]

    def _bars_of(self, name):
        if name not in self._bars:
            dates = self._engines[name].loader_.load().index.get_level_values(0)
            self._bars[name] = dates.unique()
        return self._bars[name]

    def _open_bar(self, name):
        """
        The stamp of the current bar of data source `name`, if that bar has not closed yet. A stamp
        is the start of a bar, which ends at the next stamp of its source, or for the last one
        after the bar size of its source. So data that is added up over a longer time than the
        bars of the simulation stays open while they go on. Before the bar of the simulation has
        closed time has reached its stamp, after that the next stamp of any data source.
        """
        if self._bar_closed is None:
            return None
        now = self._current_dt
        reached = now
        if self._bar_closed:
            sources = [self._bars_of(source) for source in self._engines]
            later = [bars[bars > now] for bars in sources]
            later = [bars[0] for bars in later if len(bars) > 0]
            if len(later) == 0:
                return None
            reached = min(later)

        bars = self._bars_of(name)
        following = bars.searchsorted(now, side="right")
        if following == 0:
            return None
        current = bars[following - 1]
        if following < len(bars):
            ends = bars[following]
        else:
            ends = current + (bars[1:] - bars[:-1]).min()
        return None if ends <= reached else current

    def get_price_matrix(self, length: pd.DateOffset, name: str | None = None, prices=OHLCV.close):
        """
        Prices of the last `length`, adjusted to the current bar. Until the current bar of the data
        source closes, `prices` that are only known after a close end at the bar before it, and a
        filter that reads such data decides on the bars before it.
        """
        if name is None:
            name = next(iter(self._engines))

        if name not in self._engines:
            raise KeyError(
                f"Data source '{name}' not found, please pass in the algorithm "
                f"initialization: `YourAlgorithm({name}=DataLoader())`"
            )
        engine = self._engines[name]
        start, end = self._current_dt - length, self._current_dt
        open_bar = self._open_bar(name)
        if open_bar is None:
            return engine.get_price_matrix(start, end, prices=prices)

        universe = engine.get_filter()
        delay_universe = universe is not None and universe.should_delay()
        if delay_universe:
            engine.set_filter(universe.shift(1))
        try:
            # up to `end` even if that row is dropped, a split on the current bar is known already
            matrix = engine.get_price_matrix(start, end, prices=prices)
        finally:
            if delay_universe:
                engine.set_filter(universe)
        return matrix[matrix.index < open_bar] if prices.should_delay() else matrix

    def set_datetime(self, dt: pd.Timestamp) -> None:
        self._current_dt = dt
        self.blotter.set_datetime(dt)

    @property
    def current(self):
        return self._current_dt

    @property
    def long_only(self):
        raise RuntimeError("Please using self.blotter.long_only")

    @long_only.setter
    def long_only(self, b):
        raise RuntimeError("Please using self.blotter.long_only")

    @property
    def daily_curb(self):
        raise RuntimeError("Please using self.blotter.daily_curb")

    @daily_curb.setter
    def daily_curb(self, b):
        raise RuntimeError("Please using self.blotter.daily_curb")

    @property
    def results(self):
        return self._results

    def set_history_window(self, date_offset: pd.DateOffset):
        """Set the length of historical data passed to each `rebalance` call"""
        self._history_window = date_offset

    @property
    def history_window(self):
        return self._history_window

    def record(self, **kwargs):
        self._recorder.record(self._current_dt, kwargs)

    def cumulative_returns_fig(self, annual_risk_free, benchmark, start=0):
        returns = self._results.returns
        if returns.shape[0] <= 1 or returns.isna().all():
            print("plot failed: Insufficient data")
            return

        bench = None
        if isinstance(benchmark, pd.Series):
            bench = benchmark.loc[returns.index[0] : returns.index[-1]]
        elif isinstance(benchmark, str):
            engine = self.get_factor_engine()
            filter_ = engine.get_filter()
            engine.empty_cache()
            engine.set_filter(StaticAssets({benchmark}))
            # the returns start the day before the first bar, which can be before the data
            first_bar = self._bars_of(next(iter(self._engines)))[0]
            df = engine.get_price_matrix(max(returns.index[0], first_bar), returns.index[-1])
            engine.set_filter(filter_)
            bench = df[benchmark]
            bench = bench.resample("D").last().dropna()
            if len(bench) > 2:
                bench = bench.pct_change()
            else:
                bench = None

        fig = cumulative_returns_fig(
            returns,
            self._results.positions,
            self._results.transactions,
            bench,
            annual_risk_free,
            start=start,
        )
        return fig

    def plot(self, annual_risk_free=0.04, benchmark: pd.Series | str = None) -> None:
        fig = self.cumulative_returns_fig(annual_risk_free, benchmark)
        if fig:
            fig.show()

    def _call_rebalance(self, _):
        history = self._data
        if isinstance(self._data, dict):
            if self._history_window is not None and self._history_window.n != 0:
                history = {
                    k: v.loc[(self._current_dt - self._history_window) :]
                    for k, v in self._data.items()
                }
        elif self._data is None:
            # data not ready
            return
        else:
            if self._history_window is not None and self._history_window.n != 0:
                history = self._data.loc[(self._current_dt - self._history_window) :]
        self._rebalance_callback(self._last_row, history)

    def schedule_rebalance(self, event: Event):
        """context: CustomAlgorithm.initialize"""
        self._rebalance_callback = event.callback
        event.callback = self._call_rebalance
        self.schedule(event)

    def run_engine(self, start, end, delay_factor=True):
        if start is None:
            start = self._current_dt
        start = start - self._history_window

        if len(self._engines) == 1:
            name = next(iter(self._engines))
            try:
                df = self._engines[name].run(start, end, delay_factor)
            except NoBarError:
                df = None
            if df is None or df.shape[0] == 0:
                start = start - pd.DateOffset(years=1)
                df = self._engines[name].run(start, end, delay_factor)
                return df, df.head(0)
            last_dt = df.index.get_level_values(0)[-1]
            return df, df.loc[last_dt]
        else:
            dfs = {
                name: engine.run(start, end, delay_factor) for name, engine in self._engines.items()
            }
            lasts = {k: v.loc[v.index.get_level_values(0)[-1]] for k, v in dfs.items()}
            return dfs, lasts

    def _data_updated(self, event_source=None):
        # todo if in live, last row should return by inferred time
        self._data, self._last_row = self.run_engine(None, None, delay_factor=self.delay_factor)

    def on_run(self):
        # here and not at the end of a run, the records are read after it
        self._recorder = Recorder()
        self._data = self._last_row = None
        self._bars = {}
        # `initialize` makes these again
        for name in self._engines_of_initialize:
            del self._engines[name]
        engines = set(self._engines)
        # schedule first, so it will run before rebalance
        self.schedule(EveryBarData(self._data_updated))
        try:
            self.initialize()
        finally:
            self._engines_of_initialize = self._engines.keys() - engines

    def on_end_of_run(self):
        if self.blotter.portfolio.current_dt is None:
            # stopped before the first bar, there are no results to read
            self._results = CustomAlgorithm.Results(None, None, None)
            return
        self._results = CustomAlgorithm.Results(
            returns=self.blotter.get_returns(),
            positions=self.blotter.get_historical_positions(),
            transactions=self.blotter.get_transactions(),
        )
        self.terminate(self._recorder.to_df())

    def initialize(self):
        raise NotImplementedError("abstractmethod")

    def terminate(self, records: pd.DataFrame) -> None:
        pass


# ----------------------------------------------------------------


class SimulationEventManager(EventManager):
    _mocked_data = None
    _mocked_last = None

    @classmethod
    def _get_most_granular(cls, data):
        freq = {k: min(v.index.levels[0][1:] - v.index.levels[0][:-1]) for k, v in data.items()}
        return data[min(freq, key=freq.get)]

    @classmethod
    def get_data_ticks(cls, data, start):
        if isinstance(data, dict):
            df = cls._get_most_granular(data)
            df = df.loc[start:]
        else:
            df = data[start:]
        return df.index.get_level_values(0).unique()

    @classmethod
    def wrap_data(cls, data, func):
        if isinstance(data, dict):
            return {k: func(v) for k, v in data.items()}
        else:
            return func(data)

    def fire_before_event(self, event_type):
        for events in self._subscribers.values():
            for event in events:
                if isinstance(event, event_type) and event.offset < 0:
                    event.callback(self)

    def fire_after_event(self, event_type):
        for events in self._subscribers.values():
            for event in events:
                if isinstance(event, event_type) and event.offset >= 0:
                    event.callback(self)

    def fire_market_open(self, alg):
        self.fire_before_event(MarketOpen)
        alg.blotter.set_price("open")
        alg.blotter.update_portfolio_value()
        self.fire_after_event(MarketOpen)

    def fire_market_close(self, alg):
        alg.blotter.set_price("close")
        alg.blotter.update_portfolio_value()
        alg._bar_closed = True
        self.fire_before_event(MarketClose)
        self.fire_after_event(MarketClose)

    def mock_data(self, data, datetime):
        def _mock_one(_getter):
            last = _getter.get_as_df(datetime)
            head = _getter.source.iloc[: _getter.last_row_slice.stop]
            return last, head

        if isinstance(data, dict):
            self._mocked_last = {}
            self._mocked_data = {}
            for k, getter in data.items():
                self._mocked_last[k], self._mocked_data[k] = _mock_one(getter)
        else:
            self._mocked_last, self._mocked_data = _mock_one(data)

    def run_simulation_alg(self, alg, start, end, delay_factor=True):
        from tqdm.auto import tqdm

        alg.blotter.clear()

        # get factor data from algorithm or file
        run_engine = alg.run_engine
        if "save_factor" in alg.__dict__ and os.path.exists(alg.save_factor):
            data = pd.read_feather(alg.save_factor)
            data = data.set_index(["date", "asset"])
            print("Reading factors data from file: ", alg.save_factor)
        else:
            data, _ = run_engine(start, end, delay_factor)
            if "save_factor" in alg.__dict__:
                data.reset_index().to_feather(alg.save_factor)
        # process data
        ticks = self.get_data_ticks(data, start)
        if len(ticks) == 0:
            raise ValueError("No data returned, please set `start`, `end` time correctly")
        data = self.wrap_data(data, DataLoaderFastGetter)
        # mock CustomAlgorithm
        alg.run_engine = lambda *args, **kwargs: (self._mocked_data, self._mocked_last)
        try:
            if "empty_cache_after_run" in alg.__dict__:
                for eng in alg._engines.values():
                    eng.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()

            # infer freq
            delta = min(ticks[1:] - ticks[:-1])
            data_freq = delta.resolution_string

            # for portfolio recoding cash history
            alg.blotter.portfolio.set_datetime(ticks[0] - pd.DateOffset(days=1))

            # loop factor data
            last_day = None
            for dt in tqdm(ticks):
                if self._stop:
                    break

                # if date changed
                day = dt.normalize()
                if day != last_day and last_day is not None:
                    self.fire_market_close(alg)

                # prepare data
                self.mock_data(data, dt)
                alg.set_datetime(dt)
                alg._bar_closed = False

                # fire daily data event
                if data_freq == "D":
                    self.fire_event(self, EveryBarData)

                # fire open event
                if day != last_day:
                    self.fire_market_open(alg)
                    last_day = day

                # fire intraday data event
                if data_freq != "D":
                    alg.blotter.set_price("close")
                    alg._bar_closed = True
                    self.fire_event(self, EveryBarData)

            self.fire_market_close(alg)
        finally:
            alg.run_engine = run_engine

    def run(self, start, end, delay_factor=True):
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        self._stop = False
        for r, events in self._subscribers.items():
            # clear scheduled events
            events.clear()
            if isinstance(r, CustomAlgorithm):
                r.clear()
            r.on_run()

        try:
            for alg in self._subscribers:
                if not isinstance(alg, CustomAlgorithm):
                    continue
                if not isinstance(alg.blotter, SimulationBlotter):
                    raise ValueError("SimulationEventManager only supports SimulationBlotter.")
                self.run_simulation_alg(alg, start, end, delay_factor)
        except KeyboardInterrupt:
            print("Interrupted, terminating..")

        for r in self._subscribers:
            r.on_end_of_run()
