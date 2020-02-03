"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
from zipfile import ZipFile
import numpy as np
import pandas as pd
from .dataloader import DataLoader


class QuandlLoader(DataLoader):
    @property
    def last_modified(self) -> float:
        """ the quandl data is no longer updated, so return a fixed value """
        return 1

    def __init__(self, file: str, calender_asset='AAPL') -> None:
        """
        Usage:
        download data first:
        https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?qopts.export=true&api_key=[yourapi_key]
        then:
        loader = data.QuandlLoader('./quandl/WIKI_PRICES.zip')
        """
        super().__init__(file,
                         ohlcv=('open', 'high', 'low', 'close', 'volume'),
                         adjustments=('ex-dividend', 'split_ratio'))
        self._calender = calender_asset

    def _load(self) -> pd.DataFrame:
        with ZipFile(self._path) as pkg:
            with pkg.open(pkg.namelist()[0]) as csv:
                df = pd.read_csv(csv, parse_dates=['date'],
                                 usecols=['ticker', 'date', 'open', 'high', 'low', 'close',
                                          'volume', 'ex-dividend', 'split_ratio', ],
                                 dtype={
                                     'open': np.float32, 'high': np.float32, 'low': np.float32,
                                     'close': np.float32, 'volume': np.float64,
                                     'ex-dividend': np.float64, 'split_ratio': np.float64
                                 })

        df.set_index(['date', 'ticker'], inplace=True)
        df.split_ratio.loc[("2001-09-12", 'GMT')] = 1  # fix nan
        df = self._format(df, split_ratio_is_inverse=True)
        if self._calender:
            df = self._align_to(df, self._calender)

        return df
