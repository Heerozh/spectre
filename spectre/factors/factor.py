from abc import ABC
from typing import Optional, Iterable
import pandas as pd


class BaseFactor:
    win = 1
    inputs = None

    _engine = None
    _backward = 0
    _last_output = None

    def _clean(self) -> None:
        self._backward = 0
        self._last_output = None
        if self.inputs:
            if not isinstance(self.inputs, Iterable):
                raise TypeError('`Factor.inputs` must be iterable factors.')
            for upstream in self.inputs:
                if not isinstance(upstream, BaseFactor):
                    raise TypeError('`Factor.inputs` must only contain factors.')
                upstream._clean()

    def _update_backward(self, backward=0) -> None:
        """
        Get the total win size of this tree path.
        Use to determine the start date for root Factor.
        """
        # Set the backward required by self and child, only keep max amount.
        new_backward = self.win - 1 + backward
        if new_backward > self._backward:
            self._backward = new_backward
        if self.inputs:
            for upstream in self.inputs:  # type: BaseFactor
                upstream._update_backward(self._backward)

    @classmethod
    def _create_internal_output(cls):
        """
        Create an empty internal output data structure, for `compute`
        """
        return pd.Series()

    def _compute(self, out):
        if self._last_output:
            # 如果是cuda的话，可以给cuda的数据结构做个遇到df自动转换功能
            out[:] = self._last_output[:]
            return

        # 计算所有子的数据
        if self.inputs:
            inputs = []
            for upstream in self.inputs:  # type: BaseFactor
                upstream_out = upstream._create_internal_output()
                upstream._compute(upstream_out)
                inputs.append(upstream_out)
            self.compute(out, *inputs)
        else:
            self.compute(out)
        self._last_output = out

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

    def pre_compute(self, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        pass

    def compute(self, out, *inputs) -> None:
        """
        Abstractmethod, do the actual factor calculation here.
        Unlike zipline, here calculate all data at once. Does not guarantee Look-Ahead Bias.
        Parameters
        ----------
        out
            Set to your factor value, length should be same as the inputs`
        *inputs
            All input factors data, including all data from `start(minus win)` to `end`.
        """
        raise NotImplementedError("abstractmethod")


class IndexFactor(BaseFactor, ABC):
    pass
