"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import torch
import numpy as np


class ParallelGroupBy:
    """Fast parallel group by"""

    def __init__(self, keys: torch.Tensor):
        n = keys.shape[0]
        # sort by key (keep key in GPU device)
        relative_key = keys + torch.arange(0, n, device=keys.device).double() / n  # todo test double float performance difference
        sorted_keys, sorted_indices = torch.sort(relative_key)
        sorted_keys, sorted_indices = sorted_keys.cpu().int(), sorted_indices.cpu()
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
        self._inverse_indices = inverse_indices
        self._width = width
        self._groups = groups
        self._data_shape = (groups, width)

    def split(self, data: torch.Tensor) -> torch.Tensor:
        ret = torch.take(data, self._sorted_indices)
        assert ret.type not in {torch.int8, torch.int16, torch.int32, torch.int64}, \
            'tensor cannot be any type of int, recommended to use float32'
        ret[self._sorted_indices == -1] = np.nan
        return ret

    def revert(self, split_data: torch.Tensor, dbg_str='None') -> torch.Tensor:
        if tuple(split_data.shape) != self._data_shape:
            raise ValueError('The return data shape{} of Factor `{}` must same as input{}'
                             .format(tuple(split_data.shape), dbg_str, self._data_shape))
        return torch.take(split_data, self._inverse_indices)


def nanmean(data: torch.Tensor) -> torch.Tensor:
    data = data.clone()
    isnan = torch.isnan(data)
    data[isnan] = 0
    return data.sum(dim=1) / (~isnan).sum(dim=1)


def nanstd(data: torch.Tensor) -> torch.Tensor:
    filled = data.clone()
    isnan = torch.isnan(data)
    filled[isnan] = 0
    mean = filled.sum(dim=1) / (~isnan).sum(dim=1)
    var = (data - mean[:, None]) ** 2 / (~isnan).sum(dim=1)[:, None]
    var[isnan] = 0
    return var.sum(dim=1).sqrt()


class Rolling:

    def __init__(self, x: torch.Tensor, win: int, _adj_multi: torch.Tensor = None):
        nan_stack = x.new_full((x.shape[0], win - 1), np.nan)
        new_x = torch.cat((nan_stack, x), dim=1)
        self.values = new_x.unfold(1, win, 1)
        self.win = win

        if _adj_multi is not None:
            multi = Rolling(_adj_multi, win)
            multi = multi.values / multi.values[:, :, -1, None]
            self.values = self.values * multi

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def __repr__(self):
        return 'spectre.parallel.Rolling object contains:\n' + self.values.__repr__()

    def __mul__(self, o):
        assert self.win == o.win
        r = Rolling.empty()
        r.win = self.win
        r.values = self.values * o.values
        return r

    def last(self):
        return self.values[:, :, -1]

    def first(self):
        return self.values[:, :, 0]

    def sum(self, axis=2):
        return self.values.sum(dim=axis)

    def mean(self, axis=2):
        return self.sum(axis=axis) / self.win

    def std(self, unbiased=True, axis=2):
        """
        unbiased=False eq ddof=0
        """
        return self.values.std(unbiased=unbiased, dim=axis)
