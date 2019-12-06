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

    def get_factor_engine(self, name: str = None):
        if name is None:
            name = next(iter(self._engines))

        if name not in self._engines:
            raise KeyError("Data source '{0}' not found, please pass in the algorithm "
                           "initialization: `YourAlgorithm({0}=DataLoader())`".format(name))
        return self._engines[name]

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

    def on_subscribe(self):
        # schedule first, so it will run before rebalance
        self.schedule(EveryBarData(self._run_engine))

    def initialize(self):
        raise NotImplementedError("abstractmethod")


# ----------------------------------------------------------------


class SimulationEventManager(EventManager):
    _last_day = None
    _last_date = None

    @classmethod
    def _get_most_granular(cls, data):
        freq = {k: min(v.index.levels[0][1:]-v.index.levels[0][:-1]) for k, v in data.items()}
        return data[min(freq, key=freq.get)]

    def fire_before_event(self, event_type):
        for _, events in self._subscribers.items():
            for event in events:
                if isinstance(event, event_type):
                    if event.offset < 0:
                        event.callback()

    def fire_after_event(self, event_type):
        for _, events in self._subscribers.items():
            for event in events:
                if isinstance(event, event_type):
                    if event.offset >= 0:
                        event.callback()

    def check_date_change(self, now, alg):
        # if new day
        if now.day != self._last_day:
            if self._last_day is not None:
                alg.blotter.set_price('close')
                self.fire_before_event(MarketClose)
                self.fire_after_event(MarketClose)
            alg.blotter.set_datetime(now)
            self.fire_before_event(MarketOpen)
            alg.blotter.set_price('open')
            self.fire_after_event(MarketOpen)
            self._last_day = now.day

    def run(self, start, end):
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        for r, events in self._subscribers.items():
            r.initialize()

        for alg in self._subscribers:
            if not isinstance(alg, CustomAlgorithm):
                continue
            if not isinstance(alg.blotter, SimulationBlotter):
                raise ValueError('SimulationEventManager only supports SimulationBlotter.')
            alg.blotter.clear()
            # get factor data from algorithm
            data = alg.run_engine(start, end)
            alg.run_engine = lambda x, y: self._last_date
            if isinstance(data, dict):
                main = self._get_most_granular(data)
                main = main.loc[start:end]
            else:
                main = data
            # loop factor data
            self._last_day = None
            ticks = main.index.get_level_values(0).unique()
            for dt in ticks:
                if self._stop:
                    break
                self.check_date_change(dt, alg)

                if isinstance(data, dict):
                    self._last_date = {k: v[:dt] for k, v in data.items()}
                else:
                    self._last_date = data[:dt]
                self.fire_event(self, EveryBarData)

                # todo 每tick运行完后，记录时间，然后当天new_order存到每个时间的表里

            self.fire_before_event(MarketClose)
            self.fire_after_event(MarketClose)

        for r, events in self._subscribers.items():
            r.terminate()

