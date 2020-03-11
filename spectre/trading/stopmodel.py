"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import math


def sign(x):
    return math.copysign(1, x)


class PriceTracker:
    def __init__(self, current_price, recorder=max):
        self.last_price = current_price
        self.recorder = recorder
        self.recorded_price = current_price
        self.tracking_position = None

    def update_price(self, last_price):
        self.recorded_price = self.recorder(self.recorded_price, last_price)
        self.last_price = last_price

    def process_split(self, inverse_ratio: float):
        self.recorded_price /= inverse_ratio


# -----------------------------------------------------------------------------


class StopTracker(PriceTracker):
    def __init__(self, current_price, stop_price, callback):
        super().__init__(current_price, lambda _, x: x)
        self._stop_price = stop_price
        self.stop_loss = stop_price < current_price
        self.callback = callback

    @property
    def stop_price(self):
        return self._stop_price

    def fire(self, *args):
        if callable(self.callback):
            return self.callback(*args)
        else:
            return self.callback

    def check_trigger(self, *args):
        if self.stop_loss:
            if self.last_price <= self.stop_price:
                return self.fire(*args)
        else:
            if self.last_price >= self.stop_price:
                return self.fire(*args)
        return False


class StopModel:
    def __init__(self, ratio: float, callback=None):
        self.ratio = ratio
        self.callback = callback

    def new_tracker(self, current_price, inverse):
        if inverse:
            stop_price = current_price * (1 - self.ratio)
        else:
            stop_price = current_price * (1 + self.ratio)
        return StopTracker(current_price, stop_price, self.callback)


# -----------------------------------------------------------------------------


class TrailingStopTracker(StopTracker):
    def __init__(self, current_price, ratio, callback):
        self.ratio = ratio
        stop_price = current_price * (1 + self.ratio)
        StopTracker.__init__(self, current_price, stop_price, callback=callback)
        PriceTracker.__init__(self, current_price, recorder=max if ratio < 0 else min)

    @property
    def stop_price(self):
        return self.recorded_price * (1 + self.ratio)


class TrailingStopModel(StopModel):
    """
    Unlike trailing stop order, the ratio in this model is relative to the highest / lowest price,
    so -0.1 means stop price is 90% of the highest price from now to the future; 0.1 means stop
    price is 110% of the lowest price from now to the future.
    """
    def new_tracker(self, current_price, inverse):
        ratio = -self.ratio if inverse else self.ratio
        return TrailingStopTracker(current_price, ratio, self.callback)


# -----------------------------------------------------------------------------


class DecayTrailingStopTracker(TrailingStopTracker):
    def __init__(self, current_price, ratio, target, decay_rate, max_decay, callback):
        self.initial_ratio = ratio
        self.max_decay = max_decay
        self.decay_rate = decay_rate
        self.target = target
        super().__init__(current_price, ratio, callback)

    @property
    def current(self):
        raise NotImplementedError("abstractmethod")

    @property
    def stop_price(self):
        decay = max(self.decay_rate ** (self.current / self.target), self.max_decay)
        self.ratio = self.initial_ratio * decay
        return self.recorded_price * (1 + self.ratio)


class PnLDecayTrailingStopTracker(DecayTrailingStopTracker):
    @property
    def current(self):
        pos = self.tracking_position
        pnl = (self.recorded_price / pos.average_price - 1) * sign(pos.shares)
        pnl = max(pnl, 0) if self.target > 0 else min(pnl, 0)
        return pnl


class PnLDecayTrailingStopModel(StopModel):
    """
    Exponential decay to the stop ratio: `ratio * decay_rate ^ (PnL% / PnL_target%)`.
    If it's stop gain model, `PnL_target` should be Loss Target (negative).

    So, the lower the `ratio` when PnL% approaches the target, and if PnL% exceeds PnL_target%,
    any small opposite changes will trigger stop.
    """

    def __init__(self, ratio: float, pnl_target: float, callback=None,
                 decay_rate=0.05, max_decay=0):
        super().__init__(ratio, callback)
        self.decay_rate = decay_rate
        self.pnl_target = pnl_target
        self.max_decay = max_decay

    def new_tracker(self, current_price, inverse):
        ratio = -self.ratio if inverse else self.ratio
        return PnLDecayTrailingStopTracker(
            current_price, ratio, self.pnl_target, self.decay_rate, self.max_decay, self.callback)


class TimeDecayTrailingStopTracker(DecayTrailingStopTracker):
    @property
    def current(self):
        pos = self.tracking_position
        return pos.period


class TimeDecayTrailingStopModel(StopModel):
    def __init__(self, ratio: float, period_target: 'pd.Timedelta', callback=None,
                 decay_rate=0.05, max_decay=0):
        super().__init__(ratio, callback)
        self.decay_rate = decay_rate
        self.period_target = period_target
        self.max_decay = max_decay

    def new_tracker(self, current_price, inverse):
        ratio = -self.ratio if inverse else self.ratio
        return TimeDecayTrailingStopTracker(
            current_price, ratio, self.period_target, self.decay_rate, self.max_decay,
            self.callback)
