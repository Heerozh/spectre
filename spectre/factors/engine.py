from typing import Union, Optional, Iterable
from .factor import BaseFactor, IndexFactor
import pandas as pd


class FactorEngine:
    _last_range = None

    def __init__(self) -> None:
        self._idx_factor = None
        self._factors = {}

    def _compute_factor(self, factor, out):
        # todo 跳过已计算的树
        inputs = None
        # 计算所有子的数据
        if factor.inputs:
            inputs = []
            for upstream in factor.inputs:
                upstream_out = pd.Series()
                self._compute_factor(upstream, upstream_out)
                inputs.append(upstream_out)

        factor._engine = self
        factor.compute(out, *inputs)

    def add(self, factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Optional[Union[Iterable[str], str]] = None) -> None:
        if isinstance(factor, Iterable):
            for i, fct in enumerate(factor):
                self.add(fct, name and name[i] or None)
        else:
            name = name or factor.name
            if not name:
                raise ValueError(
                    'Factor does not have `.name` attribute, '
                    'you need to specify its unique name by engine.add(factor, name).')
            name = name.format(factor.win)
            if isinstance(factor, IndexFactor):
                self._idx_factor = (name, factor)
            else:
                if name in self._factors:
                    raise KeyError('A factor with the name {} already exists.'
                                   'please specify a new name by engine.add(factor, new_name)'
                                   .format(name))
                self._factors[name] = factor

    def run(self, start: Optional[any] = None,
            end: Optional[any] = None) -> pd.DataFrame:
        if not self._idx_factor:
            raise IndexError('Need to add an `IndexFactor` before run.')

        # Calculate data that requires forward in tree
        # So each factor data range is input[start_line - forward:]
        for f in self._factors.values():
            f._clean_forward()
        for f in self._factors.values():
            f._update_forward()

        self._last_range = (start, end)

        # build df
        columns = [self._idx_factor[0], *self._factors.keys()]
        df = pd.DataFrame(columns=columns)

        # Compute factors
        self._compute_factor(self._idx_factor[1], df[self._idx_factor[0]])
        for c, f in self._factors.items():
            self._compute_factor(f, df[c])

        return df
