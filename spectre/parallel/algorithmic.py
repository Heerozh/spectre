from typing import Tuple
import torch
import numpy as np


def groupby(data: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, list, dict]:
    """Fast parallel group by"""
    n = data.shape[0]
    # sort by key
    relative_key = key + torch.arange(0, n, device=data.device).float() / n
    sorted_keys, sorted_key_indices = torch.sort(relative_key)
    # get group
    diff = sorted_keys[1:].int() - sorted_keys[:-1].int()
    boundary = (diff.nonzero(as_tuple=True)[0] + 1).cpu().tolist()
    boundary = np.array([0] + boundary + [n])
    width = max(boundary[1:] - boundary[:-1])
    # split
    groups = len(boundary) - 1
    ret = data.new_zeros((groups, width))
    inverse_indices, key_to_group = [], {}
    for start, end, i in zip(boundary[:-1], boundary[1:], range(groups)):
        ret[i, 0:(end-start)] = data[sorted_key_indices[start:end]]
        key_to_group[int(sorted_keys[start].item())] = i
        inverse_indices.append(sorted_key_indices[start:end])
    return ret, inverse_indices, key_to_group



