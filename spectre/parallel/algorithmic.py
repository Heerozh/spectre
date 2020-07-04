"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from typing import Callable
import torch
import numpy as np
import warnings
from .constants import DeviceConstant
from ..config import Global


class ParallelGroupBy:
    """Fast parallel group by"""
    GROUPBY_GPU_SORT_MAX_MEMORY = 200 * 1024 ** 2

    def __init__(self, keys: torch.Tensor):
        assert keys.min() >= 0
        n = keys.shape[0]
        # sort by key (keep key in GPU device)
        relative_key = keys + torch.linspace(0, 0.9, n, dtype=torch.double, device=keys.device)
        sorted_keys, sorted_indices = torch.sort(relative_key)
        sorted_keys, sorted_indices = sorted_keys.int(), sorted_indices.cpu()
        # get group boundary
        diff = sorted_keys[1:] - sorted_keys[:-1]
        boundary = (diff.nonzero(as_tuple=True)[0] + 1).tolist()
        boundary = np.array([0] + boundary + [n])
        del relative_key, sorted_keys, diff
        # get inverse indices
        width = np.diff(boundary).max()
        groups = len(boundary) - 1
        inverse_indices = sorted_indices.new_full((groups, width), n + 1)
        for start, end, i in zip(boundary[:-1], boundary[1:], range(groups)):
            inverse_indices[i, 0:(end - start)] = sorted_indices[start:end]
        # keep inverse_indices in GPU for sort
        inverse_indices_memory = inverse_indices.element_size() * inverse_indices.nelement()
        if inverse_indices_memory > self.GROUPBY_GPU_SORT_MAX_MEMORY:
            warnings.warn('Groupby uses too much memory, switching to CPU for sorting, may affect '
                          'initialization performance or modify spectre.parallel.ParallelGroupBy.'
                          'GROUPBY_GPU_SORT_MAX_MEMORY parameter',
                          RuntimeWarning)
            inverse_indices = torch.sort(inverse_indices)[1][:n]
            inverse_indices = inverse_indices.flatten().to(keys.device, non_blocking=True)
        else:
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

    @property
    def padding_mask(self):
        return self._padding_mask

    def split(self, data: torch.Tensor) -> torch.Tensor:
        ret = torch.take(data, self._sorted_indices)
        if ret.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
            print(data)
            raise ValueError('tensor cannot be any type of int, recommended to use float32')
        if ret.dtype == torch.bool:
            ret.masked_fill_(self._padding_mask, False)
        else:
            ret.masked_fill_(self._padding_mask, np.nan)
        return ret

    def revert(self, split_data: torch.Tensor, dbg_str='None') -> torch.Tensor:
        if tuple(split_data.shape) != self._data_shape:
            if tuple(split_data.shape[:2]) == self._data_shape[:2]:
                raise ValueError('The downstream needs shape{2}, and the input factor "{1}" is '
                                 'shape{0}. Look like this factor has multiple return values, '
                                 'using slice to select a value before using it, for example: '
                                 '`factor[0]`.'
                                 .format(tuple(split_data.shape), dbg_str, self._data_shape))
            else:
                raise ValueError('The return data shape{} of Factor `{}` must same as input{}.'
                                 .format(tuple(split_data.shape), dbg_str, self._data_shape))
        return torch.take(split_data, self._inverse_indices)

    def create(self, dtype, values, nan_fill=np.nan):
        ret = self._sorted_indices.new_full(self._sorted_indices.shape, values, dtype=dtype)
        ret.masked_fill_(self._padding_mask, nan_fill)
        return ret


def unmasked_sum(data: torch.Tensor, mask: torch.Tensor, dim=1) -> torch.Tensor:
    data = data.clone()
    data.masked_fill_(mask, 0)  # much faster than data[isnan] = 0
    return data.sum(dim=dim)


def unmasked_prod(data: torch.Tensor, mask: torch.Tensor, dim=1) -> torch.Tensor:
    data = data.clone()
    data.masked_fill_(mask, 1)
    return data.prod(dim=dim)


def nansum(data: torch.Tensor, dim=1) -> torch.Tensor:
    mask = torch.isnan(data)
    return unmasked_sum(data, mask, dim)


def nanprod(data: torch.Tensor, dim=1) -> torch.Tensor:
    mask = torch.isnan(data)
    return unmasked_prod(data, mask, dim)


def unmasked_mean(data, mask, dim=1):
    total = unmasked_sum(data, mask, dim)
    return total / (~mask).sum(dim=dim)


def nanmean(data: torch.Tensor, dim=1) -> torch.Tensor:
    mask = torch.isnan(data)
    return unmasked_mean(data, mask, dim)


def unmasked_var(data: torch.Tensor, mask, dim=1, ddof=0) -> torch.Tensor:
    total = unmasked_sum(data, mask, dim)
    n = (~mask).sum(dim=dim)
    mean = total / n
    mean.unsqueeze_(-1)
    var = (data - mean) ** 2 / (n.unsqueeze(-1) - ddof)
    var.masked_fill_(mask, 0)
    return var.sum(dim=dim)


def nanvar(data: torch.Tensor, dim=1, ddof=0) -> torch.Tensor:
    mask = torch.isnan(data)
    return unmasked_var(data, mask, dim, ddof)


def nanstd(data: torch.Tensor, dim=1, ddof=0) -> torch.Tensor:
    return nanvar(data, dim, ddof).sqrt()


def nanmax(data: torch.Tensor, dim=1) -> torch.Tensor:
    data = data.clone()
    isnan = torch.isnan(data)
    data.masked_fill_(isnan, -np.inf)
    return data.max(dim=dim)[0]


def nanmin(data: torch.Tensor, dim=1) -> torch.Tensor:
    data = data.clone()
    isnan = torch.isnan(data)
    data.masked_fill_(isnan, np.inf)
    return data.min(dim=dim)[0]


def masked_last(data: torch.Tensor, mask: torch.Tensor, dim=1, reverse=False) -> torch.Tensor:
    if reverse:
        w = DeviceConstant.get(data.device).r_linspace(mask.shape[-1], dtype=torch.float)
    else:
        w = DeviceConstant.get(data.device).linspace(mask.shape[-1], dtype=torch.float)

    w = mask.float() + w
    last = w.argmax(dim=dim)
    ret = data.gather(dim, last.unsqueeze(-1)).squeeze(-1)
    ret_mask = mask.gather(dim, last.unsqueeze(-1)).squeeze(-1)
    ret = torch.masked_fill(ret, ~ret_mask, np.nan)
    return ret


def masked_first(data: torch.Tensor, mask: torch.Tensor, dim=1) -> torch.Tensor:
    return masked_last(data, mask, dim, reverse=True)


def nanlast(data: torch.Tensor, dim=1, offset=0) -> torch.Tensor:
    if offset > 0:
        s = [slice(None)] * len(data.shape)
        s[dim] = slice(offset, None)
        data = data[s]
    mask = ~torch.isnan(data)
    return masked_last(data, mask, dim)


def pad_2d(data: torch.Tensor, including_inf=False) -> torch.Tensor:
    mask = torch.isnan(data)
    if including_inf:
        mask = mask | torch.isinf(data)
    idx = torch.arange(0, mask.shape[1], device=data.device).expand(mask.shape[0], mask.shape[1])
    idx = idx.masked_fill(mask, 0)
    idx = idx.cummax(dim=1).values
    return torch.gather(data, 1, idx)


def rankdata(data: torch.Tensor, dim=1, ascending=True, method='average', normalize=False):
    nans = torch.isnan(data)
    if not ascending:
        filled = data.masked_fill(nans, -np.inf)
    else:
        filled = data
    arr, sorter = torch.sort(filled, dim=dim, descending=not ascending)
    rank, inv = torch.sort(sorter.to(torch.int32), dim=dim)
    del sorter

    if method == 'ordinal':
        ret = (inv.to(Global.float_type) + 1.)
    else:
        flt = DeviceConstant.get(inv.device).arange(np.prod(inv.shape[:dim]), dtype=inv.dtype)
        inv += (flt * inv.shape[dim]).view(*inv.shape[:dim], 1)

        obs = arr != arr.roll(1, dims=dim)
        del arr

        if method == 'average':
            lower = rank.masked_fill(~obs, 0)
            upper = rank.masked_fill(~obs.roll(-1, dims=dim), rank.shape[dim]).flip(dims=[dim])
            del rank, obs
            lower = lower.cummax(dim=dim).values
            upper = upper.cummin(dim=dim).values.flip(dims=[dim])

            avg = (upper + lower + 2) * .5
            ret = torch.take(avg, inv)
        elif method == 'dense':
            dense = obs.cumsum(dim=1).to(Global.float_type)
            ret = torch.take(dense, inv)

    if normalize:
        ret /= data.shape[dim]
    ret.masked_fill_(nans, np.nan)
    return ret


def unmasked_covariance(x, y, mask, dim=1, ddof=0):
    x = x - unmasked_mean(x, dim=dim, mask=mask).unsqueeze(-1)
    y = y - unmasked_mean(y, dim=dim, mask=mask).unsqueeze(-1)
    xy = x * y
    xy.masked_fill_(mask, 0)
    xy = xy.sum(dim=dim)
    return xy / ((~mask).sum(dim=dim) - ddof)


def covariance(x, y, dim=1, ddof=0):
    mask = torch.isnan(x * y)
    return unmasked_covariance(x, y, mask, dim, ddof)


def pearsonr(x, y, dim=1, ddof=0):
    mask = torch.isnan(x * y)
    cov = unmasked_covariance(x, y, mask, dim, ddof)
    x_var = unmasked_var(x, mask, dim, ddof)
    y_var = unmasked_var(y, mask, dim, ddof)
    return cov / (x_var.sqrt() * y_var.sqrt())


def linear_regression_1d(x, y, dim=1):
    x_bar = nanmean(x, dim=dim).unsqueeze(-1)
    y_bar = nanmean(y, dim=dim).unsqueeze(-1)
    demean_x = x - x_bar
    demean_y = y - y_bar
    cov = nanmean(demean_x * demean_y, dim=dim)
    x_var = nanvar(x, dim=dim, ddof=0)
    slope = cov / x_var
    slope[x_var == 0] = 0
    intcp = y_bar.squeeze() - slope * x_bar.squeeze()
    return slope, intcp


def quantile(data, bins, dim=1):
    if data.dtype == torch.bool:
        data = data.char()
    if data.shape[1] == 1:  # if only one asset in universe
        return data.new_full(data.shape, 0, dtype=Global.float_type)

    x, _ = torch.sort(data, dim=dim)
    # get non-nan size of each row
    mask = torch.isnan(data)
    act_size = data.shape[dim] - mask.sum(dim=dim)
    # get each bin's cut indices of each row by non-nan size
    q = torch.linspace(0, 1, bins + 1, device=data.device)
    q = q.view(-1, *[1 for _ in range(dim)])
    q_index = q * (act_size - 1)
    # calculate un-perfect cut weight
    q_weight = q % 1
    q_index = q_index.long()
    q_next = q_index + 1
    q_next[-1] = act_size - 1

    # get quantile values of each row
    dim_len = data.shape[dim]
    offset = torch.arange(0, q_index[0].nelement(), device=data.device) * dim_len
    offset = offset.reshape(q_index[0].shape)
    q_index += offset
    q_next += offset
    b_start = x.take(q_index)
    b_end = x.take(q_next)
    b = b_start + (b_end - b_start) * q_weight
    b[0] -= 1
    b = b.unsqueeze(-1)

    ret = data.new_full(data.shape, np.nan, dtype=Global.float_type)
    for start, end, tile in zip(b[:-1], b[1:], range(bins)):
        ret[(data > start) & (data <= end)] = tile
    return ret


def masked_kth_value_1d(data, mask, k_array, treat_nan_as=0):
    # fill nan with 0, and fill out of mask with inf
    data = data.masked_fill(~mask, np.inf)
    nans = torch.isnan(data)
    data.masked_fill_(nans, treat_nan_as)
    # sort
    sorted_data, _ = data.sort(dim=1)
    # gather median
    value = sorted_data.gather(1, k_array)
    return value, sorted_data


def clamp_1d_(data, min_, max_):
    min_mask = data < min_
    max_mask = data > max_
    min_ = min_.expand(min_.shape[0], data.shape[1])
    max_ = max_.expand(max_.shape[0], data.shape[1])

    data.masked_scatter_(min_mask, min_.masked_select(min_mask))
    data.masked_scatter_(max_mask, max_.masked_select(max_mask))
    return data


class Rolling:
    _split_multi = 1  # 0.5-1 recommended, you can tune this for kernel performance

    @classmethod
    def unfold(cls, x, win, fill=np.nan):
        nan_stack = x.new_full((x.shape[0], win - 1), fill)
        new_x = torch.cat((nan_stack, x), dim=1)
        return new_x.unfold(1, win, 1)

    def __init__(self, x: torch.Tensor, win: int, _adjustment: torch.Tensor = None):
        self.values = self.unfold(x, win)
        self.win = win

        # rolling multiplication will consume lot of memory, split it by size
        memory_usage = self.values.nelement() * win / (1024. ** 3)
        memory_usage *= Rolling._split_multi
        step = max(int(self.values.shape[1] / memory_usage), 1)
        boundary = list(range(0, self.values.shape[1], step)) + [self.values.shape[1]]
        self.split = list(zip(boundary[:-1], boundary[1:]))

        if _adjustment is not None:
            rolling_adj = Rolling(_adjustment, win)
            self.adjustments = rolling_adj.values
            self.adjustment_last = rolling_adj.last_nonnan()[:, :, None]
        else:
            self.adjustments = None
            self.adjustment_last = None

    def adjust(self, s=None, e=None) -> torch.Tensor:
        """this will contiguous tensor consume lot of memory, limit e-s size"""
        if self.adjustments is not None:
            return self.values[:, s:e] * self.adjustments[:, s:e] / self.adjustment_last[:, s:e]
        else:
            return self.values[:, s:e]

    def __repr__(self):
        return 'spectre.parallel.Rolling object contains:\n' + self.values.__repr__()

    def agg(self, op: Callable, *others: 'Rolling'):
        """
        Call `op` on the split rolling data one by one, pass in all the adjusted values,
        and finally aggregate them into a whole.
        """
        assert all(r.win == self.win for r in others), '`others` must have same `win` with `self`'
        seq = [op(self.adjust(s, e), *[r.adjust(s, e) for r in others]).contiguous()
               for s, e in self.split]
        return torch.cat(seq, dim=1)

    def loc(self, i):
        if i == -1:
            # last doesn't need to adjust, just return directly
            return self.values[:, :, i]

        def _loc(x):
            return x[:, :, i]

        return self.agg(_loc)

    def last(self):
        return self.loc(-1)

    def last_nonnan(self, offset=0):
        return self.agg(lambda x: nanlast(x, dim=2, offset=offset))

    def first(self):
        return self.loc(0)

    def sum(self, axis=2):
        return self.agg(lambda x: x.sum(dim=axis))

    def nansum(self, axis=2):
        return self.agg(lambda x: nansum(x, dim=axis))

    def nanprod(self, axis=2):
        return self.agg(lambda x: nanprod(x, dim=axis))

    def mean(self, axis=2):
        return self.agg(lambda x: x.sum(dim=axis) / self.win)

    def nanmean(self, axis=2):
        return self.agg(lambda x: nanmean(x, dim=axis))

    def std(self, axis=2):
        # unbiased=False eq ddof=0
        return self.agg(lambda x: x.std(unbiased=False, dim=axis))

    def nanstd(self, axis=2):
        return self.agg(lambda x: nanstd(x, dim=axis, ddof=0))

    def var(self, axis=2):
        return self.agg(lambda x: x.var(unbiased=False, dim=axis))

    def nanvar(self, axis=2):
        return self.agg(lambda x: nanvar(x, dim=axis, ddof=0))

    def max(self):
        return self.agg(lambda x: x.max(dim=2)[0])

    def min(self):
        return self.agg(lambda x: x.min(dim=2)[0])

    def nanmax(self):
        return self.agg(lambda x: nanmax(x, dim=2))

    def nanmin(self):
        return self.agg(lambda x: nanmin(x, dim=2))
