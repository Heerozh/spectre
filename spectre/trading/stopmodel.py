"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""


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

    def new_tracker(self, current_price):
        return StopTracker(current_price, current_price * (1 + self.ratio), self.callback)


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
    def new_tracker(self, current_price):
        return TrailingStopTracker(current_price, self.ratio, self.callback)


class DecayTrailingStopTracker(TrailingStopTracker):
    """Exponential decay to the stop ratio: ratio * decay_rate ^ (PnL% / PnL_target)"""

    def __init__(self, current_price, ratio, decay_rate, pnl_target, callback):
        self.fixed_ratio = ratio
        self.decay_rate = decay_rate
        self.pnl_target = pnl_target
        super().__init__(current_price, ratio, callback)

    @property
    def stop_price(self):
        pnl = self.tracking_position.unrealized_percent
        pnl = max(pnl, 0) if self.pnl_target > 0 else min(pnl, 0)
        self.ratio = self.fixed_ratio * (self.decay_rate ** (pnl / self.pnl_target))
        return self.recorded_price * (1 + self.ratio)


class DecayTrailingStopModel(StopModel):
    def __init__(self, ratio: float, callback=None, decay_rate=0.05, pnl_target=0.1):
        super().__init__(ratio, callback)
        self.decay_rate = decay_rate
        self.pnl_target = pnl_target

    def new_tracker(self, current_price):
        return DecayTrailingStopTracker(current_price, self.ratio,
                                        self.decay_rate, self.pnl_target,self.callback)
