from typing import Tuple, List
import torch
import numpy as np


# ----------- group -----------

class ParallelGroupBy:
    """Fast parallel group by"""
    def __init__(self, key: torch.Tensor):
        n = key.shape[0]
        # sort by key
        relative_key = key + torch.arange(0, n, device=key.device).double() / n  # todo test double float performance difference
        sorted_keys, sorted_indices = torch.sort(relative_key)
        # get group boundary
        diff = sorted_keys[1:].int() - sorted_keys[:-1].int()
        boundary = (diff.nonzero(as_tuple=True)[0] + 1).cpu().tolist()
        boundary = np.array([0] + boundary + [n])
        # get inverse indices
        width = max(boundary[1:] - boundary[:-1])
        groups = len(boundary) - 1
        inverse_indices = sorted_indices.new_full((groups, width), n+1)
        for start, end, i in zip(boundary[:-1], boundary[1:], range(groups)):
            inverse_indices[i, 0:(end-start)] = sorted_indices[start:end]
        inverse_indices = torch.sort(inverse_indices.flatten())[1][:n]
        # class members
        self._boundary = boundary
        self._sorted_indices = sorted_indices
        self._inverse_indices = inverse_indices
        self._width = width
        self._groups = groups
        self._data_shape = (groups, width)

    def split(self, data: torch.Tensor) -> torch.Tensor:
        # split
        ret = data.new_zeros((self._groups, self._width))
        for start, end, i in zip(self._boundary[:-1], self._boundary[1:], range(self._groups)):
            ret[i, 0:(end - start)] = data[self._sorted_indices[start:end]]
        return ret

    def revert(self, split_data: torch.Tensor, dbg_str=None) -> torch.Tensor:
        if tuple(split_data.shape) != self._data_shape:
            raise ValueError('The return data shape{} of Factor `{}` must same as input{}'
                             .format(tuple(split_data.shape), dbg_str, self._data_shape))
        return torch.take(split_data, self._inverse_indices)


class Rolling:

    def __init__(self, x: torch.Tensor, win: int):
        nan_stack = x.new_full((x.shape[0], win-1), np.nan)
        new_x = torch.cat((nan_stack, x), dim=1)
        self.x = new_x.unfold(1, win, 1)
        self.win = win

    def sum(self, axis=2):
        return self.x.sum(dim=axis)
