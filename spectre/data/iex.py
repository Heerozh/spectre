"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import os
import numpy as np
from .dataloader import ArrowLoader, CsvDirLoader
from .iex_fetcher import iex


class IexDownloader:  # todo IexDownloader unfinished
    @classmethod
    def _concat(cls):
        pass

    @classmethod
    def ingest(cls, iex_key, save_to, range_='5y', symbols: list = None, skip_exists=True):
        """
        Download data from IEX. Please note that downloading all the data will cost around $60.
        :param iex_key: your private api key of IEX account.
        :param save_to: path to folder
        :param range_: historical range, supports 5y, 2y, 1y, ytd, 6m, 3m, 1m.
        :param symbols: list of symbol to download. If is None, download All Stocks
                        (not including delisted).
        :param skip_exists: skip if file exists, useful for resume from interruption.
        """
        from tqdm.auto import tqdm
        print("Download prices from IEX...")
        iex.init(iex_key, api='cloud')

        calender_asset = None
        if symbols is None:
            symbols = iex.Reference.symbols()
            types = (symbols.type == 'ad') | (symbols.type == 'cs') & (symbols.exchange != 'OTC')
            symbols = symbols[types].symbol.values
            symbols.extend(['SPY', 'QQQ'])
            calender_asset = 'SPY'

        def download(event, folder):
            for symbol in tqdm(symbols):
                csv_path = os.path.join(folder, '{}.csv'.format(symbol))
                if os.path.exists(csv_path) and skip_exists:
                    continue
                if event == 'chart':
                    iex.Stock(symbol).chart(range_).to_csv(csv_path)
                elif event == 'dividends':
                    iex.Stock(symbol).dividends(range_).to_csv(csv_path)
                elif event == 'splits':
                    iex.Stock(symbol).splits(range_).to_csv(csv_path)

        print('Ingest prices...')
        prices_dir = os.path.join(save_to, 'daily')
        if not os.path.exists(prices_dir):
            os.makedirs(prices_dir)
        download('chart', prices_dir)

        print('Ingest dividends...')
        div_dir = os.path.join(save_to, 'dividends')
        if not os.path.exists(div_dir):
            os.makedirs(div_dir)
        download('dividends', div_dir)

        print('Ingest splits...')
        sp_dir = os.path.join(save_to, 'splits')
        if not os.path.exists(sp_dir):
            os.makedirs(sp_dir)
        download('splits', sp_dir)

        print('Converting...')
        use_cols = {'date', 'uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume', 'exDate', 'amount',
                    'ratio'}
        loader = CsvDirLoader(
            prices_dir, calender_asset=calender_asset,
            dividends_path=div_dir, splits_path=sp_dir,
            ohlcv=('uOpen', 'uHigh', 'uLow', 'uClose', 'uVolume'), adjustments=('amount', 'ratio'),
            prices_index='date', dividends_index='exDate', splits_index='exDate',
            parse_dates=True, usecols=lambda x: x in use_cols,
            dtype={'uOpen': np.float32, 'uHigh': np.float32, 'uLow': np.float32,
                   'uClose': np.float32,
                   'uVolume': np.float64, 'amount': np.float64, 'ratio': np.float64})

        arrow_file = os.path.join(save_to, 'yahoo.feather')
        ArrowLoader.ingest(source=loader, save_to=arrow_file, force=True)

        print('Ingest completed! Use `loader = spectre.data.ArrowLoader(r"{}")` '
              'to load your data.'.format(arrow_file))

    @classmethod
    def update(cls, range_, temp_path, save_to):
        # todo download and concat
        pass
