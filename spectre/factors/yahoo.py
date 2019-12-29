"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import datetime
import os
import time
import pandas as pd
import numpy as np

from .dataloader import ArrowLoader, CsvDirLoader


class YahooDownloader:

    @classmethod
    def ingest(cls, start_date: datetime, save_to, skip_exists=True) -> None:
        """
        Download all SPY stock from yahoo, Including dividends and split data.
        Yahoo will not like this, use carefully.
        """
        import requests
        import re
        from tqdm.auto import tqdm

        print("Download all SPY stock from yahoo, Including dividends and split data."
              "Yahoo will not like this, use carefully.")

        start_date = pd.to_datetime(start_date, utc=True)

        etf = pd.read_html(requests.get(
            'https://etfdailynews.com/etf/spy/', headers={'User-agent': 'Mozilla/5.0'}
        ).text, attrs={'id': 'etfs-that-own'})
        symbols = [x for x in etf[0].Symbol.values.tolist() if isinstance(x, str)]
        symbols.extend(['SPY', 'QQQ'])

        session = requests.Session()
        page = session.get('https://finance.yahoo.com/quote/IBM/history?p=IBM')
        # CrumbStore
        m = re.search('"CrumbStore":{"crumb":"(.*?)"}', page.text)
        crumb = m.group(1)
        crumb = crumb.encode('ascii').decode('unicode-escape')

        def download(event, folder):
            start = int(start_date.timestamp())
            now = int(datetime.datetime.now().timestamp())
            for symbol in tqdm(symbols):
                symbol = symbol.replace('.', '-')
                csv_path = os.path.join(folder, '{}.csv'.format(symbol))
                if os.path.exists(csv_path) and skip_exists:
                    continue
                url = "https://query1.finance.yahoo.com/v7/finance/download/" \
                      "{}?period1={}&period2={}&interval=1d&events={}&crumb={}".format(
                       symbol, start, now, event, crumb)

                retry = 0.25
                while True:
                    req = session.get(url)
                    if req.status_code != requests.codes.ok:
                        if 'No data found' in req.text:
                            print('Symbol invalid, skipped: {}.'.format(symbol))
                            break
                        retry *= 2
                        if retry >= 5:
                            print('Get {} failed, Over 4 retries, skipped, reason: {}'.format(
                                symbol, req.text))
                            break
                        else:
                            time.sleep(retry)
                            continue
                    with open(csv_path, 'wb') as f:
                        f.write(req.content)
                    break

        print('Ingest prices...')
        prices_dir = os.path.join(save_to, 'daily')
        if not os.path.exists(prices_dir):
            os.makedirs(prices_dir)
        download('history', prices_dir)

        # yahoo prices data already split adjusted
        # print('Ingest dividends...')
        # div_dir = os.path.join(save_to, 'dividends')
        # if not os.path.exists(div_dir):
        #     os.makedirs(div_dir)
        # download('div', div_dir)

        # print('Ingest splits...')
        # sp_dir = os.path.join(save_to, 'splits')
        # if not os.path.exists(sp_dir):
        #     os.makedirs(sp_dir)
        # download('split', sp_dir)

        print('Converting...')
        loader = CsvDirLoader(
            prices_dir, calender_asset='SPY',
            # dividends_path=div_dir,
            # splits_path=sp_dir,
            ohlcv=('Open', 'High', 'Low', 'Close', 'Volume'),
            # adjustments=('Dividends', 'Stock Splits'),
            prices_index='Date',
            # dividends_index='Date', splits_index='Date', split_ratio_is_fraction=True,
            parse_dates=True,
            dtype={'Open': np.float32, 'High': np.float32, 'Low': np.float32,
                   'Close': np.float32,
                   'Volume': np.float64, 'Dividends': np.float64})

        arrow_file = os.path.join(save_to, 'yahoo.feather')
        ArrowLoader.ingest(source=loader, save_to=arrow_file, force=True)

        print('Ingest completed! Use `loader = spectre.factors.ArrowLoader(r"{}")` '
              'to load your data.'.format(arrow_file))
