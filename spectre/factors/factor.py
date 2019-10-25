from abc import ABC
from typing import Optional, Iterable
import pandas as pd


class BaseFactor:
    win = 1
    inputs = None
    name = None

    _engine = None
    _forward = 0

    def _clean_forward(self) -> None:
        self._forward = 0
        if self.inputs:
            if not isinstance(self.inputs, Iterable):
                raise TypeError('`Factor.inputs` must be iterable factors.')
            for upstream in self.inputs:
                if not isinstance(upstream, BaseFactor):
                    raise TypeError('`Factor.inputs` must only contain factors.')
                upstream._clean_forward()

    def _update_forward(self, forward=0) -> None:
        """
        Get the total win size of this tree path.
        Use to determine the start date for root Factor.
        """
        # Set the forward required by self and child, only keep max amount.
        new_forward = self.win - 1 + forward
        if new_forward > self._forward:
            self._forward = new_forward
        if self.inputs:
            for upstream in self.inputs:  # type: BaseFactor
                upstream._update_forward(self._forward)

    def __init__(self, win: Optional[int] = None,
                 inputs: Optional[Iterable[object]] = None) -> None:
        """
        Parameters
        ----------
        win : Optional[int]
            Including additional past data with 'window length' in `input`
            when passed to the `compute` function.
            If not specified, use `self.win` instead.
        inputs: Optional[Iterable[BaseFactor]]
            Input factors, will all passed to the `compute` function.
            If not specified, use `self.inputs` instead.
        """
        if win:
            self.win = win
        if inputs:
            self.inputs = inputs

    def compute(self, out: pd.Series, *inputs) -> None:
        """
        Abstractmethod, do the actual factor calculation here.
        Unlike zipline, here calculate all data at once.
        Parameters
        ----------
        out : pd.Series
            Set to your factor value, length should be same with the `len(input[start:])`
        *inputs
            All input factors data, including all data from `start(minus win)` to `end`.
        """
        raise NotImplementedError("abstractmethod")


class IndexFactor(BaseFactor, ABC):
    pass
