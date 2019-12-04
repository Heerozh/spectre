"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from abc import ABC
from .event import *
from .blotter import BaseBlotter
from ..factors import FactorEngine
from ..factors import DataLoader


class BaseAlgorithm(EventReceiver, ABC):
    """
    Base class for custom trading algorithm.
    """
    def __init__(self, blotter: BaseBlotter, **data_sources: DataLoader):
        """
        :param blotter: order management system for this algorithm.
        :param data_sources: key is data_source_name, value is dataloader
        """
        super().__init__()
        if data_sources:
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
        event.callback = lambda x: origin_callback(self._data)
        self.schedule(event)

    def create_data(self, start, end):
        if len(self._engines) == 1:
            name = next(iter(self._engines))
            return self._engines[name].run_last(start, end)
        else:
            return {name: engine.run_last(start, end) for name, engine in self._engines.items()}

    def _data_updated(self):
        self._data = self.create_data(None, None)

    def on_subscribe(self):
        # schedule first, so it will run before rebalance
        self.schedule(EveryBarData(self._data_updated))

    def analyze(self):
        pass


# ----------------------------------------------------------------


class SimulationEventManager(EventManager):
    _last_day = None

    @classmethod
    def _get_most_granular(cls, data):
        freq = {k: min(v.index.levels[0][1:]-v.index.levels[0][:-1]) for k, v in data.items()}
        return data[min(freq, key=freq.get)]

    @classmethod
    def fire_before_event(cls, events, event_type):
        for event in events:
            if isinstance(event, event_type):
                if event.offset < 0:
                    event.callback()

    @classmethod
    def fire_after_event(cls, events, event_type):
        for event in events:
            if isinstance(event, event_type):
                if event.offset >= 0:
                    event.callback()

    def fire_market_event(self, now, events):
        # if new day
        if now.day != self._last_day:
            if self._last_day is not None:
                self.fire_before_event(events, MarketClose)
                self.fire_after_event(events, MarketClose)
            self.fire_before_event(events, MarketOpen)
            self.fire_after_event(events, MarketOpen)
            self._last_day = now.day

    def run(self, start, end):
        if not self._subscribers:
            raise ValueError("At least one subscriber.")

        for r, events in self._subscribers.items():
            r.initialize()

        for r, events in self._subscribers.items():
            # get factor data from algorithm
            data = r.create_data(start, end)
            if isinstance(data, dict):
                main = self._get_most_granular(data)
                main = main[start:end]
            else:
                main = data
            # loop factor data
            self._last_day = None
            ticks = main.index.get_level_values(0).unique()
            for now in ticks:
                if self._stop:
                    break
                self.fire_market_event(now, events)

                if isinstance(data, dict):
                    r._data = {k: v[:now] for k, v in data.items()}
                else:
                    r._data = data[:now]
                self.fire_event(EveryBarData)

                # todo 每tick运行完后，记录时间，然后当天new_order存到每个时间的表里

            self.fire_before_event(events, MarketClose)
            self.fire_after_event(events, MarketClose)



