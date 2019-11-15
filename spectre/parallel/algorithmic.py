import torch
import numpy as np

# ----------- group -----------

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
        inverse_indices = sorted_indices.new_full((groups, width), n+1).pin_memory()
        for start, end, i in zip(boundary[:-1], boundary[1:], range(groups)):
            inverse_indices[i, 0:(end-start)] = sorted_indices[start:end]
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

    def __init__(self, x: torch.Tensor, win: int):
        nan_stack = x.new_full((x.shape[0], win-1), np.nan)
        new_x = torch.cat((nan_stack, x), dim=1)
        self.x = new_x.unfold(1, win, 1)
        self.win = win

    def sum(self, axis=2):
        return self.x.sum(dim=axis)

    def mean(self, axis=2):
        return self.sum(axis=axis) / self.win

    def std(self, unbiased=True, axis=2):
        """
        unbiased=False eq ddof=0
        """
        return self.x.std(unbiased=unbiased, dim=axis)
