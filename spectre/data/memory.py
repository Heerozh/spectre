"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from .dataloader import DataLoader


class MemoryLoader(DataLoader):
    """ Convert pd.Dataframe to spectre.data.DataLoader """
    def __init__(self, df, ohlcv=None, adjustments=None) -> None:
        super().__init__("", ohlcv, adjustments)
        self.df = self._format(df)
        self.test_load()

    @property
    def last_modified(self) -> float:
        return 1

    def _load(self):
        return self.df
