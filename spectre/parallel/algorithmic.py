"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Callable
import torch
import numpy as np


class ParallelGroupBy:
    """Fast parallel group by"""

    def __init__(self, keys: torch.Tensor):
        n = keys.shape[0]
        # sort by key (keep key in GPU device)
        relative_key = keys + torch.arange(0, n, device=keys.device).double() / n
        sorted_keys, sorted_indices = torch.sort(relative_key)
        sorted_keys, sorted_indices = sorted_keys.int(), sorted_indices.cpu()
        # get group boundary
        diff = sorted_keys[1:] - sorted_keys[:-1]
        boundary = (diff.nonzero(as_tuple=True)[0] + 1).tolist()
        boundary = np.array([0] + boundary + [n])
        # get inverse indices
        width = max(boundary[1:] - boundary[:-1])
        groups = len(boundary) - 1
        inverse_indices = sorted_indices.new_full((groups, width), n + 1).pin_memory()
        for start, end, i in zip(boundary[:-1], boundary[1:], range(groups)):
            inverse_indices[i, 0:(end - start)] = sorted_indices[start:end]
        # keep inverse_indices in GPU for sort
        inverse_indices = inverse_indices.flatten().to(keys.device, non_blocking=True)
        inverse_indices = torch.sort(inverse_indices)[1][:n]
        # for fast split
        take_indices = sorted_indices.new_full((groups, width), -1)
        for start, end, i in zip(boundary[:-1], boundary[1:], range(groups)):
            take_indices[i, 0:(end - start)] = sorted_indices[start:end]
        take_indices = take_indices.to(keys.device, non_blocking=True)
        # class members
        self._boundary = boundary
        self._sorted_indices = take_indices
        self._padding_mask = take_indices == -1
        self._inverse_indices = inverse_indices
        self._width = width
        self._groups = groups
        self._data_shape = (groups, width)

    def split(self, data: torch.Tensor) -> torch.Tensor:
        ret = torch.take(data, self._sorted_indices)
        assert ret.type not in {torch.int8, torch.int16, torch.int32, torch.int64}, \
            'tensor cannot be any type of int, recommended to use float32'
        ret.masked_fill_(self._padding_mask, np.nan)
        return ret

    def revert(self, split_data: torch.Tensor, dbg_str='None') -> torch.Tensor:
        if tuple(split_data.shape) != self._data_shape:
            raise ValueError('The return data shape{} of Factor `{}` must same as input{}'
                             .format(tuple(split_data.shape), dbg_str, self._data_shape))
        return torch.take(split_data, self._inverse_indices)

    def create(self, dtype, values, nan_values):
        ret = self._sorted_indices.new_full(self._sorted_indices.shape, values, dtype=dtype)
        ret.masked_fill_(self._padding_mask, np.nan)
        return ret


def nanmean(data: torch.Tensor, dim=1) -> torch.Tensor:
    data = data.clone()
    isnan = torch.isnan(data)
    data[isnan] = 0
    return data.sum(dim=dim) / (~isnan).sum(dim=dim)


def nanstd(data: torch.Tensor, dim=1) -> torch.Tensor:
    filled = data.clone()
    isnan = torch.isnan(data)
    filled[isnan] = 0
    mean = filled.sum(dim=dim) / (~isnan).sum(dim=dim)
    mean.unsqueeze_(-1)
    var = (data - mean) ** 2 / (~isnan).sum(dim=dim).unsqueeze(-1)
    var[isnan] = 0
    return var.sum(dim=dim).sqrt()


def nanlast(data: torch.Tensor, dim=1) -> torch.Tensor:
    mask = torch.isnan(data)
    w = torch.linspace(0.1, 0, mask.shape[-1], dtype=torch.float, device=mask.device)
    mask = mask.float() + w
    last = mask.argmin(dim=dim)
    return data.gather(dim, last.unsqueeze(-1)).squeeze()


class Rolling:
    _split_multi = 64  # 32-64 recommended, you can tune this for kernel performance

    @classmethod
    def unfold(cls, x, win, fill=np.nan):
        nan_stack = x.new_full((x.shape[0], win - 1), fill)
        new_x = torch.cat((nan_stack, x), dim=1)
        return new_x.unfold(1, win, 1)

    def __init__(self, x: torch.Tensor, win: int, _adjustment: torch.Tensor = None):
        self.values = self.unfold(x, win)
        self.win = win

        # rolling multiplication will consume lot of memory, split it by size
        memory_usage = self.values.nelement() / (1024. ** 3)
        memory_usage *= Rolling._split_multi
        step = int(self.values.shape[0] / memory_usage)
        boundary = list(range(0, self.values.shape[0], step)) + [self.values.shape[0]]
        self.split = list(zip(boundary[:-1], boundary[1:]))

        if _adjustment is not None:
            rolling_adj = Rolling(_adjustment, win)
            self.adjustment = rolling_adj.values
            self.adjustment_last = rolling_adj.last_nonnan()[:, :, None]
        else:
            self.adjustment = None
            self.adjustment_last = None

    def adjusted(self, s=None, e=None) -> torch.Tensor:
        """this will contiguous tensor consume lot of memory, limit e-s size"""
        if self.adjustment is not None:
            return self.values[s:e] * self.adjustment[s:e] / self.adjustment_last[s:e]
        else:
            return self.values[s:e]

    def __repr__(self):
        return 'spectre.parallel.Rolling object contains:\n' + self.values.__repr__()

    def agg(self, op: Callable, *others: 'Rolling'):
        """
        Call `op` on the split rolling data one by one, pass in all the adjusted values,
        and finally aggregate them into a whole.
        """
        assert all(r.win == self.win for r in others), '`others` must have same `win` with `self`'
        seq = [op(self.adjusted(s, e), *[r.adjusted(s, e) for r in others]) for s, e in self.split]
        return torch.cat(seq)

    def loc(self, i):
        if i == -1:
            # last doesn't need to adjust, just return directly
            return self.values[:, :, i]

        def _loc(x):
            return x[:, :, i]

        return self.agg(_loc)

    def last(self):
        return self.loc(-1)

    def last_nonnan(self):
        return self.agg(lambda x: nanlast(x, dim=2))

    def first(self):
        return self.loc(0)

    def sum(self, axis=2):
        def _sum(x):
            return x.sum(dim=axis)

        return self.agg(_sum)

    def mean(self, axis=2):
        def _mean(x):
            # return nanmean(x, dim=axis)
            # tensor.sum encounters nan will return nan anyway
            return x.sum(dim=axis) / self.win

        return self.agg(_mean)

    def std(self, axis=2):
        def _std(x):
            # return nanstd(x, dim=axis)
            # unbiased=False eq ddof=0
            return x.std(unbiased=False, dim=axis)

        return self.agg(_std)

    def max(self):
        def _max(x):
            return x.max(dim=2)[0]

        return self.agg(_max)

    def min(self):
        def _min(x):
            return x.min(dim=2)[0]

        return self.agg(_min)
