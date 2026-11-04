"""
Small helpers to keep spectre working across pandas versions.
"""
import pandas as pd

PANDAS_GE_3 = int(pd.__version__.split('.')[0]) >= 3


def stack_keep_nan(df: pd.DataFrame, level=-1):
    """
    `df.stack(level)` that keeps all-NaN rows, same as the old `df.stack(level, dropna=False)`.
    `dropna` was removed in pandas 3.0, the new implementation never drops rows.
    """
    if PANDAS_GE_3:
        return df.stack(level)
    return df.stack(level, future_stack=True)
