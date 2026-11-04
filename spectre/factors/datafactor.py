"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from ..config import Global
from ..parallel import nanlast
from .factor import BaseFactor, CustomFactor


class ColumnDataFactor(BaseFactor):
    def __init__(self, inputs: Sequence[str] | None = None, should_delay=True, dtype=None) -> None:
        super().__init__()
        if inputs:
            self.inputs = inputs
        assert 3 > len(self.inputs) > 0, (
            "ColumnDataFactor's `inputs` can only contains one data column and corresponding "
            "adjustments column"
        )
        self._data = None
        self._multiplier = None
        self._should_delay = should_delay
        self.dtype = dtype

    @property
    def adjustments(self):
        return self._multiplier

    def get_total_backwards_(self) -> int:
        return 0

    def should_delay(self) -> bool:
        return self._should_delay

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        if self._data is None:
            self._data = engine.column_to_tensor_(self.inputs[0])
            if self.dtype is not None:
                self._data = self._data.to(self.dtype)
            self._data = engine.group_by_(self._data, self.groupby)
            if len(self.inputs) > 1 and self.inputs[1] in engine.dataframe_:
                self._multiplier = engine.column_to_tensor_(self.inputs[1])
                self._multiplier = engine.group_by_(self._multiplier, self.groupby)
                if self.dtype is not None:
                    self._multiplier = self._multiplier.to(self.dtype)
            else:
                self._multiplier = None
            self._clean_required = True

    def clean_up_(self, force=False) -> None:
        super().clean_up_(force)
        self._data = None
        self._multiplier = None
        self._clean_required = False

    def compute_(self, stream: torch.cuda.Stream | None) -> torch.Tensor:
        return self._data

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        pass


#     def adjusted_shift(self, periods=1):
#         factor = AdjustedShiftFactor(win=periods, inputs=(self,))
#         return factor
#
#
# class AdjustedShiftFactor(CustomFactor):
#     """ Shift the root datafactor """
#
#     def compute(self, data) -> torch.Tensor:
#         return data.first()


class AdjustedColumnDataFactor(CustomFactor):
    def __init__(self, data: ColumnDataFactor):
        super().__init__(1, (data,))
        self.parent = data

    def compute(self, data) -> torch.Tensor:
        multi = self.parent.adjustments
        if multi is None:
            return data
        return data * multi / nanlast(multi, dim=1)[:, None]


class AssetClassifierDataFactor(BaseFactor):
    """Dict to categorical output for asset, slow"""

    def __init__(self, sector: dict, default: int):
        super().__init__()
        self.sector = sector
        self.default = default
        self._data = None

    def get_total_backwards_(self) -> int:
        return 0

    def should_delay(self) -> bool:
        return False

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        assets = engine.dataframe_index[1]
        sector = self.sector
        default = self.default
        data = [sector.get(asset, default) for asset in assets]  # slow
        data = torch.tensor(data, device=engine.device, dtype=Global.float_type)
        self._data = engine.group_by_(data, self.groupby)

    def clean_up_(self, force=False) -> None:
        super().clean_up_(force)
        self._data = None

    def compute_(self, stream: torch.cuda.Stream | None) -> torch.Tensor:
        return self._data

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        pass


class SeriesDataFactor(ColumnDataFactor):
    """
    Add series to engine, slow. `fill_na` fills each row from the values of its asset in the whole
    series, so also from dates before the loaded rows or without a bar: 'ffill' or 'pad' takes the
    last value up to the row, 'bfill' or 'backfill' the next value from the row.
    """

    _fill_directions = {
        "ffill": "backward",
        "pad": "backward",
        "bfill": "forward",
        "backfill": "forward",
    }

    def __init__(self, series: pd.Series, fill_na=None, should_delay=True):
        self.series = series
        self.fill_method = fill_na
        assert series.index.names == ["date", "asset"], (
            "df.index.names should be ['date', 'asset'] "
        )
        super().__init__(inputs=[str(series.name)], should_delay=should_delay)
        self._aligned = None
        self._aligned_hash = None

    def _align(self, engine) -> np.ndarray:
        """The value of the series for each row of the frame."""
        if self.fill_method is not None and self.fill_method not in self._fill_directions:
            raise ValueError(f"`fill_na` must be one of {list(self._fill_directions)}")
        if self.series.index.has_duplicates:
            raise ValueError("The series is non-unique, a row can't hold 2 values.")
        if self.fill_method is None:
            return self.series.reindex(engine.dataframe_.index).to_numpy()

        dates, assets = engine.dataframe_index
        index = self.series.index
        # by the labels of the level, a code of -1 is a NaN label, so it gets the -1 at the end
        asset_codes = np.append(assets.categories.get_indexer(index.levels[1]), -1)[index.codes[1]]
        values = pd.DataFrame(
            {
                "date": index.get_level_values(0).tz_convert(dates.tz).as_unit(dates.unit),
                "asset": asset_codes,
                "value": self.series.to_numpy(),
            }
        )
        rows = pd.DataFrame({"date": dates, "asset": assets.codes.astype(np.int64)})
        return pd.merge_asof(
            rows,
            values.dropna().sort_values("date", kind="stable"),
            on="date",
            by="asset",
            direction=self._fill_directions[self.fill_method],
        )["value"].to_numpy()

    def pre_compute_(self, engine, start, end) -> None:
        # Not a column of the frame. Its name says nothing about its data, pandas keeps the name on
        # derived series, and a private column would stay in the frame after the run, one for each
        # series ever run on it. What it read is kept here, for the next run on these rows.
        BaseFactor.pre_compute_(self, engine, start, end)
        if self._data is None:
            if self._aligned_hash != engine.cache_hash:
                self._aligned = self._align(engine)
                self._aligned_hash = engine.cache_hash
            self._data = engine.group_by_(self._aligned, self.groupby)
            self._clean_required = True


class DatetimeDataFactor(BaseFactor):
    """Datetime's attr to DataFactor"""

    _instance = {}
    # deprecated attributes of DatetimeIndex
    _aliases = {
        "weekday": "day_of_week",
        "dayofweek": "day_of_week",
        "dayofyear": "day_of_year",
        "daysinmonth": "days_in_month",
    }

    def __new__(cls, attr):
        if attr not in cls._instance:
            cls._instance[attr] = super().__new__(cls)
        return cls._instance[attr]

    def __init__(self, attr) -> None:
        super().__init__()
        self._data = None
        self.attr = attr

    def get_total_backwards_(self) -> int:
        return 0

    def should_delay(self) -> bool:
        return False

    def pre_compute_(self, engine, start, end) -> None:
        super().pre_compute_(engine, start, end)
        if self._data is None:
            attr = self._aliases.get(self.attr, self.attr)
            data = getattr(engine.dataframe_index[0], attr)  # slow
            if not isinstance(data, np.ndarray):
                data = data.values
            if not data.flags.writeable:
                data = data.copy()
            data = torch.from_numpy(data).to(
                device=engine.device, dtype=Global.float_type, non_blocking=True
            )
            self._data = engine.group_by_(data, self.groupby)
            self._clean_required = True

    def clean_up_(self, force=False) -> None:
        super().clean_up_(force)
        self._data = None
        self._clean_required = False

    def compute_(self, stream: torch.cuda.Stream | None) -> torch.Tensor:
        return self._data

    def compute(self, *inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        pass
