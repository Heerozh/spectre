from typing import Optional, Sequence


class BaseFactor:
    def _get_total_backward(self) -> int:
        raise NotImplementedError("abstractmethod")

    def _pre_compute(self, engine, start, end) -> None:
        raise NotImplementedError("abstractmethod")

    def _compute(self) -> any:
        raise NotImplementedError("abstractmethod")

    def __init__(self, win: Optional[int] = None,
                 inputs: Optional[Sequence[any]] = None) -> None:
        pass

    def compute(self, *inputs) -> any:
        raise NotImplementedError("abstractmethod")


class CustomFactor(BaseFactor): #todo 还要继承有top，加减等方法的term
    win = 1
    inputs = None

    _min_win = None
    _cache = None
    _cache_hit = 0

    def _get_total_backward(self) -> int:
        backward = 0
        if self.inputs:
            backward = max([up._get_total_backward()
                            for up in self.inputs if isinstance(up, BaseFactor)] or (0,))
        return backward + self.win - 1

    def _pre_compute(self, engine, start, end) -> None:
        """
        Called when engine run but before compute.
        """
        self._cache = None
        self._cache_hit = 0
        if self.inputs:
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream._pre_compute(engine, start, end)

    def _compute(self) -> any:
        if self._cache is not None:
            self._cache_hit += 1
            return self._cache

        # Calculate inputs
        if self.inputs:
            inputs = []
            for upstream in self.inputs:
                if isinstance(upstream, BaseFactor):
                    upstream_out = upstream._compute()
                    inputs.append(upstream_out)
                else:
                    inputs.append(upstream)
            out = self.compute(*inputs)
        else:
            out = self.compute()
        self._cache = out
        return out

    def __init__(self, win: Optional[int] = None, inputs: Optional[Sequence[BaseFactor]] = None):
        """
        :param win:  Optional[int]
            Including additional past data with 'window length' in `input`
            when passed to the `compute` function.
            **If not specified, use `self.win` instead.**
        :param inputs: Optional[Iterable[OHLCV|BaseFactor]]
            Input factors, will all passed to the `compute` function.
            **If not specified, use `self.inputs` instead.**
        """
        super().__init__(win, inputs)
        if win:
            self.win = win
        if inputs:
            self.inputs = inputs

        assert (self.win >= (self._min_win or 1))

    def compute(self, *inputs) -> any:
        """
        Abstractmethod, do the actual factor calculation here.
        Unlike zipline, here calculate all data at once. Does not guarantee Look-Ahead Bias.
        Data structure:
                | index    | stock1 | ... | stockN |
                |----------|--------|-----|--------|
                | datetime | input  | ... | input  |
        :param inputs: All input factors data, including all data from `start(minus win)` to `end`.
        :return: your factor values, length should be same as the inputs`
        """
        raise NotImplementedError("abstractmethod")


class DataFactor(BaseFactor):
    def _get_total_backward(self) -> int:
        return 0

    def __init__(self, inputs: Optional[Sequence[str]] = None) -> None:
        super().__init__(inputs)
        if inputs:
            self.inputs = inputs

    def _pre_compute(self, engine, start, end) -> None:
        df = engine.get_loader_data()
        self._data = df[self.inputs[0]].unstack(level=1)

    def _compute(self) -> any:
        return self._data

    def compute(self, *inputs) -> any:
        pass


class FilterFactor(BaseFactor):
    pass
