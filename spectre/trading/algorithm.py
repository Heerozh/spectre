"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import pandas as pd
from abc import ABC
from .event import *
from .blotter import BaseBlotter, SimulationBlotter
from ..factors import FactorEngine
from ..factors import DataLoader


class Record:
    def __init__(self):
        self._records = []

    def record(self, date, table):
        if 'date' in table:
            raise ValueError('`date` is reserved key for record.')
        table['date'] = date
        self._records.append(table)

    def to_df(self):
        ret = pd.DataFrame(self._records)
        if ret.shape[0] > 0:
            ret = ret.set_index('date').sort_index(axis=0)
        return ret


class CustomAlgorithm(EventReceiver, ABC):
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

        self._data = None
        self._engines = {name: FactorEngine(loader) for name, loader in data_sources.items()}
        self.blotter = blotter
        self._records = Record()
        self._current_dt = None

    def clear(self):
        for engine in self._engines.values():
            engine.clear()

    def get_factor_engine(self, name: str = None):
        if name is None:
            name = next(iter(self._engines))

        if name not in self._engines:
            raise KeyError("Data source '{0}' not found, please pass in the algorithm "
                           "initialization: `YourAlgorithm({0}=DataLoader())`".format(name))
        return self._engines[name]

    def set_datetime(self, dt: pd.Timestamp) -> None:
        self._current_dt = dt
        self.blotter.set_datetime(dt)

    @property
    def current(self):
        return self._current_dt

    def record(self, **kwargs):
        self._records.record(self._current_dt, kwargs)

    def schedule_rebalance(self, event: Event):
        """Can only be called in initialize()"""
        origin_callback = event.callback

        def _rebalance_callback(_):
            if isinstance(self._data, dict):
                last = {k: v.loc[v.index.get_level_values(0)[-1]]
                        for k, v in self._data.items()}
            else:
                last_dt = self._data.index.get_level_values(0)[-1]
                last = self._data.loc[last_dt]
            origin_callback(last, self._data)
        event.callback = _rebalance_callback
        self.schedule(event)

    def run_engine(self, start, end):
        if len(self._engines) == 1:
            name = next(iter(self._engines))
            return self._engines[name].run(start, end)
        else:
            return {name: engine.run(start, end) for name, engine in self._engines.items()}

    def _run_engine(self, event_source=None):
        self._data = self.run_engine(None, None)

    def on_run(self):
        # schedule first, so it will run before rebalance
        self.schedule(EveryBarData(self._run_engine))
        self.initialize()

    def on_end_of_run(self):
        self.terminate(self._records.to_df())

    def initialize(self):
        raise NotImplementedError("abstractmethod")

    def terminate(self, records: pd.DataFrame) -> None:
        pass


# ----------------------------------------------------------------


class SimulationEventManager(EventManager):
    _last_data = None

    @classmethod
    def _get_most_granular(cls, data):
        freq = {k: min(v.index.levels[0][1:]-v.index.levels[0][:-1]) for k, v in data.items()}
        return data[min(freq, key=freq.get)]

    def fire_before_event(self, event_type):
        for _, events in self._subscribers.items():
            for event in events:
                if isinstance(event, event_type):
                    if event.offset < 0:
                        event.callback(self)

    def fire_after_event(self, event_type):
        for _, events in self._subscribers.items():
            for event in events:
                if isinstance(event, event_type):
                    if event.offset >= 0:
                        event.callback(self)

    def fire_market_open(self, alg):
        self.fire_before_event(MarketOpen)
        alg.blotter.set_price('open')
        alg.blotter.update_portfolio_value()
        self.fire_after_event(MarketOpen)

    def fire_market_close(self, alg):
        alg.blotter.set_price('close')
        alg.blotter.update_portfolio_value()
        self.fire_before_event(MarketClose)
        self.fire_after_event(MarketClose)

    def run(self, start, end):
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        for r, events in self._subscribers.items():
            # clear scheduled events
            events.clear()
            if isinstance(r, CustomAlgorithm):
                r.clear()
            r.on_run()

        for alg in self._subscribers:
            if not isinstance(alg, CustomAlgorithm):
                continue
            if not isinstance(alg.blotter, SimulationBlotter):
                raise ValueError('SimulationEventManager only supports SimulationBlotter.')
            alg.blotter.clear()
            # get factor data from algorithm
            data = alg.run_engine(start, end)
            alg.run_engine = lambda x, y: self._last_data
            if isinstance(data, dict):
                main = self._get_most_granular(data)
                main = main.loc[start:end]
            else:
                main = data
            # loop factor data
            last_day = None
            ticks = main.index.get_level_values(0).unique()
            for dt in ticks:
                if self._stop:
                    break
                # prepare data
                if isinstance(data, dict):
                    self._last_data = {k: v[:dt] for k, v in data.items()}
                else:
                    self._last_data = data[:dt]

                # if date changed
                if dt.day != last_day:
                    if last_day is not None:
                        self.fire_market_close(alg)
                    alg.set_datetime(dt)

                # fire daily data event
                if dt.hour == 0:
                    self.fire_event(self, EveryBarData)

                # fire open event
                if dt.day != last_day:
                    self.fire_market_open(alg)
                    last_day = dt.day

                # fire intraday data event
                if dt.hour != 0:
                    self.fire_event(self, EveryBarData)

            self.fire_market_close(alg)

        for r in self._subscribers.keys():
            r.on_end_of_run()

