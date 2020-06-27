"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2019-2020, Heerozh. All rights reserved.
@license: Apache 2.0
@email: heeroz@gmail.com
"""
import io
import os
from struct import unpack, calcsize
import re

import pandas as pd
import numpy as np
import glob
from tqdm.auto import tqdm

from spectre.data.dataloader import DataLoader
from spectre.trading import CNCalendar


class TDXLoader(DataLoader):
    """ China stock daily price loader """

    def __init__(self, tdx_vipdoc_dir: str, start=None,
                 universe=('^SH60.*', '^SZ00.*', 'SH000001'),
                 calender_asset='SH000001', align_by_time=False) -> None:
        """
            universe 可以是：
                大盘： ' SZ399001', 'SZ399006', 'SZ399300', 'SZ399905'
                创业板: ^SZ30.*',
        """
        super().__init__(tdx_vipdoc_dir,
                         ohlcv=('open', 'high', 'low', 'close', 'volume'),
                         adjustments=('ex-dividend', 'split_ratio'))
        self._calender = calender_asset
        self.universe = [re.compile(m) for m in universe]
        self._align_by_time = align_by_time
        self.start = start

    @property
    def last_modified(self) -> float:
        pattern = os.path.join(self._path, "*.*")
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError("Dir '{}' does not contains any files.".format(self._path))
        return max([os.path.getmtime(fn) for fn in files])

    def _load(self) -> pd.DataFrame:
        def tick_in_universe(_ticker_):
            for m in self.universe:
                if re.match(m, _ticker_):
                    return True
            return False

        def read_daily_price(file_handle, price_divisor):
            data = []
            while True:
                binary = file_handle.read(4 * 5 + 4 * 3)
                if not binary:
                    break
                row = unpack('<IIIIIfII', binary)
                data.append(row)
            ret = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'turnover',
                                              'volume', 'unused'])
            price_cols = ['open', 'high', 'low', 'close']
            ret[price_cols] /= price_divisor
            ret[price_cols] = ret[price_cols].astype(np.float32)
            ret['volume'] = ret['volume'].astype(np.float64)
            ret['date'] = pd.to_datetime(ret['date'].astype(str), format='%Y%m%d')
            ret.set_index('date', inplace=True)
            ret.index = ret.index.tz_localize('Asia/Shanghai')
            ret.drop(columns=['unused'], inplace=True)
            if self.start is not None:
                ret = ret.loc[self.start:]
            return ret[~ret.index.duplicated(keep='last')]

        def read_adjustment(file_handle):
            file_handle.seek(104, 1)
            date = unpack('<I', file_handle.read(4))
            rows = []
            while date[0] != 0xFFFFFFFF:
                row = unpack('<ffff100s', file_handle.read(0x74))
                rows.append(date + row)

                next_date = file_handle.read(4)
                if not next_date:
                    break
                date = unpack('<I', next_date)
            ret = pd.DataFrame(rows, columns=['date', 'split', 'allotment', 'allotment_price',
                                              'dividend', 'note'])
            ret['date'] = pd.to_datetime(ret['date'], unit='s')
            ret['note'] = ret['note'].str.decode("gbk").str.rstrip('\0')
            ret.set_index('date', inplace=True)
            ret.index = ret.index.tz_localize('Asia/Shanghai')
            return ret

        def read_fundamentals(file_handle):
            mem = file_handle.read()
            header_pack_format = '<1hI1H3L'
            header_size = calcsize(header_pack_format)
            stock_item_size = calcsize("<6s1c1L")
            data_header = mem[0:header_size]
            stock_header = unpack(header_pack_format, data_header)
            max_count = stock_header[2]
            report_date = stock_header[1]
            report_size = stock_header[4]
            report_fields_count = int(report_size / 4)
            report_pack_format = '<{}f'.format(report_fields_count)

            selected = []
            for stock_idx in range(0, max_count):
                cur = header_size + stock_idx * calcsize("<6s1c1L")
                si = mem[cur:cur + stock_item_size]
                stock_item = unpack("<6s1c1L", si)
                code = stock_item[0].decode("utf-8")
                foa = stock_item[2]
                cur = foa

                info_data = mem[cur:cur + calcsize(report_pack_format)]
                cw_info = unpack(report_pack_format, info_data)
                selected.append((report_date, code, cw_info[0], cw_info[3], cw_info[5],
                                 cw_info[237], cw_info[238],))

            ret = pd.DataFrame(selected, columns=['date', 'asset', 'eps', 'bvps', 'roe',
                                                  'outstanding', 'floating'])
            ret['date'] = pd.to_datetime(ret['date'].astype(str), format='%Y%m%d') \
                .dt.tz_localize('Asia/Shanghai')
            ret['roe'] = ret['roe'].astype(np.float32)
            ret['eps'] = ret['eps'].astype(np.float32)
            ret['bvps'] = ret['bvps'].astype(np.float32)
            ret.asset = ret.asset.str.replace(r'^00', 'SZ00')
            ret.asset = ret.asset.str.replace(r'^60', 'SH60')
            ret.set_index(['date', 'asset'], inplace=True)
            return ret

        print('Loading prices...')

        exchanges = ['sh', 'sz']

        # read all daily prices
        price_dfs = {}
        for exchange in exchanges:
            pattern = os.path.join(self._path, exchange, 'lday', "*.day")
            files = glob.glob(pattern)
            for fn in tqdm(files):
                ticker = os.path.basename(fn)[:-4].upper()
                if not tick_in_universe(ticker):
                    continue
                with io.open(fn, 'rb') as f:
                    price_divisor = 100
                    # 基金，债券，逆回购等除数是1000，这里只考虑了基金
                    if ticker.startswith('SH5') or ticker.startswith('SZ1'):
                        price_divisor = 1000
                    df = read_daily_price(f, price_divisor)
                    price_dfs[ticker] = df

        print('Loading adjustments...')

        # read all div splits
        adj_dfs = {}
        for exchange in exchanges:
            filename = os.path.join(self._path, 'full_{}.PWR'.format(exchange))
            with io.open(filename, 'rb') as f:
                f.seek(12)
                while True:
                    ticker = f.read(12)
                    if not ticker:
                        break
                    ticker = ticker.decode("utf-8").rstrip('\0').upper()
                    adj = read_adjustment(f)
                    if tick_in_universe(ticker):
                        adj_dfs[ticker] = adj

        print('Merging prices and adjustments...')

        # merge adjustments to prices
        for ticker in tqdm(price_dfs.keys()):
            df = price_dfs[ticker]
            if ticker not in adj_dfs:
                df['ex-dividend'] = 0
                df['split_ratio'] = 1
                price_dfs[ticker] = df
                continue
            # 配股相当于支付给对方分红，支付price*allotment
            # 注意，这个算法逻辑上问题是不大的，但是会导致涨停板计算错误，好在大股一般没配股的问题
            # 所以涨停板之类，得使用交易所的除权方式自己重新计算，或者用交易量成交/滑点模型实现
            adjustment = adj_dfs[ticker]
            ex_div = adjustment['dividend']
            ex_div -= adjustment['allotment_price'] * adjustment['allotment']
            ex_div = ex_div.reindex(df.index)
            ex_div = ex_div.fillna(0)
            ex_div.name = 'ex-dividend'
            sp_rto = adjustment['split'] + 1 + adjustment['allotment']
            sp_rto = sp_rto.reindex(df.index)
            sp_rto = sp_rto.fillna(1)
            sp_rto.name = 'split_ratio'
            price_dfs[ticker] = pd.concat([df, ex_div, sp_rto], axis=1)

        ret_df = pd.concat(price_dfs, sort=False)
        ret_df = ret_df.rename_axis(['asset', 'date'])
        ret_df = ret_df.swaplevel(0, 1).sort_index()

        print('Extend date to next trading day (for live trading) ...')
        # infer next trading day, and add nan rows:
        delta = min(ret_df.index.levels[0][1:] - ret_df.index.levels[0][:-1])
        unit = delta.resolution_string
        freq = int(delta / pd.Timedelta(1, unit))
        if unit == 'D':
            unit = 'B'
        last = ret_df.index.levels[0][-1]
        next_date = CNCalendar(pop_passed=True).events['Open'][0].normalize()
        new_section = ret_df.xs(last, drop_level=False).copy()
        new_section[:] = np.nan
        new_section['ex-dividend'] = 0
        new_section['split_ratio'] = 1
        new_section.index = new_section.index.remove_unused_levels()
        new_section.index = new_section.index.set_levels([next_date], level=0)
        ret_df = ret_df.append(new_section)

        # formatting
        print('Formatting data...')
        if self._calender:
            ret_df = self._align_to(ret_df, self._calender, self._align_by_time)
            if self._align_by_time:
                ret_df['ex-dividend'] = ret_df['ex-dividend'].fillna(0)
                ret_df['split_ratio'] = ret_df['split_ratio'].fillna(1)
            ret_df.sort_index(level=[0, 1], inplace=True)
        ret_df = self._format(ret_df, split_ratio_is_inverse=True)

        print('Loading fundamentals...')
        fund_dfs = []
        pattern = os.path.join(self._path, 'cw', "gpcw*.dat")
        files = glob.glob(pattern)
        for fn in tqdm(files):
            with io.open(fn, 'rb') as f:
                df = read_fundamentals(f)
                fund_dfs.append(df)
        fund_df = pd.concat(fund_dfs)
        fund_df = fund_df[~fund_df.index.duplicated(keep='last')]
        fund_df.sort_index(level=0, inplace=True)

        print('Merging prices and fundamentals...')
        ret_df = ret_df.join(fund_df)
        cols = fund_df.columns
        ret_df[cols] = ret_df[cols].groupby(level=1).fillna(method='pad')

        print('Merging prices and Index weights...')
        pattern = os.path.join(self._path, "*_weights.csv")
        files = glob.glob(pattern)
        for fn in files:
            name = os.path.basename(fn)[:-12].upper()
            df = pd.read_csv(fn, parse_dates=['effDate'])
            df.consID = df.consID.str.replace(r'(.*)(.XSHG)', r'SH\1')
            df.consID = df.consID.str.replace(r'(.*)(.XSHE)', r'SZ\1')
            df = df.set_index(['effDate', 'consID'])

            s = df.weight / 100
            s = s.astype(np.float32)
            # 对于移出的设为0，为了方便之后的pad na
            s = s.unstack().stack(dropna=False).fillna(0)
            s.index = s.index.set_names(['date', 'asset'])
            s.name = name

            ret_df = ret_df.join(s)
            # 填充Nan
            ret_df[name] = ret_df[name].groupby(level=1).fillna(method='pad')
            # 把0权重的设为nan
            ret_df[name] = ret_df[name].replace(to_replace=0, value=np.nan)
            print(name, ' weights OK.')

        print('Merging sector data...')
        # 使用从申万主页下载的申万一级行业数据
        html_dfs = pd.read_html(os.path.join(self._path, "SwClass"), encoding='GBK',
                                converters={'股票代码': str})
        sector_df = html_dfs[0]
        sector_df['股票代码'] = sector_df['股票代码'].str.replace(r'^00(.*)', r'SZ00\1')
        sector_df['股票代码'] = sector_df['股票代码'].str.replace(r'^30(.*)', r'SZ30\1')
        sector_df['股票代码'] = sector_df['股票代码'].str.replace(r'^60(.*)', r'SH60\1')
        sector_df['股票代码'] = sector_df['股票代码'].str.replace(r'^68(.*)', r'SH68\1')
        sector_df = sector_df.set_index('股票代码')
        # unique_sector = sector_df['行业名称'].unique()
        # sector_cat = dict(zip(unique_sector, range(len(unique_sector))))
        # sector_dict = sector_df['行业名称'].map(sector_cat).to_dict()
        # ret_df['sector'] = ret_df.index.map(lambda x: sector_dict.get(x[1], -1))
        # ret_df['sector'] = ret_df['sector'].astype('int32')
        name_dict = sector_df['股票名称'].to_dict()
        ret_df['name'] = ret_df.index.map(lambda x: name_dict.get(x[1], 'UNKNOWN'))
        # 使用uqer的申万行业历史数据
        sector = pd.read_csv(os.path.join(self._path, "sector.csv"), encoding='utf',
                             parse_dates=['date'])
        sector['asset'] = sector['asset'].str.replace(r'(.*)(.XSHG)', r'SH\1')
        sector['asset'] = sector['asset'].str.replace(r'(.*)(.XSHE)', r'SZ\1')
        sector.date = pd.to_datetime(sector.date, utc=True)
        sector = sector.sort_values(by=['date', 'asset'])
        # 还是不用他的name，因为申万的st标志更及时
        sector = sector.drop(columns=['name'])

        cols, new_cols = ['l1', 'l2', 'l3'], ['sector_l1', 'sector_l2', 'sector_l3']
        for lv in cols:
            unique_lv = sector[lv].unique()
            lv_cat = dict(zip(unique_lv, range(len(unique_lv))))
            sector[lv] = sector[lv].map(lv_cat)
            sector[lv] = sector[lv].astype('int32')

        sector.rename(columns={**dict(zip(cols, new_cols)), 'asset': 'str_asset'},
                      inplace=True)

        ret_df = ret_df.reset_index()
        ret_df['str_asset'] = ret_df.asset.astype('str')
        ret_df = pd.merge_asof(ret_df, sector, on='date', by='str_asset')
        ret_df = ret_df.set_index(['date', 'asset']).sort_index(level=[0, 1]).\
            drop(columns=['str_asset'])
        # ret_df.name = ret_df.name.fillna('UNKNOWN')
        ret_df[new_cols] = ret_df[new_cols].fillna(-1)

        print('Merging ST tag...')
        st_df = pd.read_csv(os.path.join(self._path, "ST_tag.csv"), parse_dates=['tradeDate'])
        st_df.secID = st_df.secID.str.replace(r'(.*)(.XSHG)', r'SH\1')
        st_df.secID = st_df.secID.str.replace(r'(.*)(.XSHE)', r'SZ\1')
        st_df = st_df.set_index(['tradeDate', 'secID'])
        st_value, st_value_index = pd.factorize(st_df.STflg, sort=True)
        st_df.STflg = st_value + 1
        st_df.STflg = st_df.STflg.astype(np.float)

        st_s = st_df.STflg
        st_s.name = 'st_tag'
        st_s.index = st_s.index.set_names(['date', 'asset'])
        ret_df = ret_df.join(st_s)

        # st数据影响幅度较大，所以Nan的行按最近的股票名来设定是否ST，注意有前视偏差
        for tag in ['ST', 'SST', '*ST', 'S*ST', 'S', '*']:
            mask = ret_df['name'].str.startswith(tag)
            ret_df.st_tag = ret_df.st_tag.mask(
                mask, ret_df.st_tag.fillna(st_value_index.searchsorted(tag) + 1))
        ret_df.st_tag = ret_df.st_tag.fillna(0)

        print('Merging ShortAble Tag...')
        short_able_df = pd.read_csv(os.path.join(
            self._path, "short_able.csv"), dtype={'stockCode': str, 'converRate': int},
            encoding='gbk')
        short_able_df['asset'] = (short_able_df['marketCode'] + short_able_df['stockCode'])\
            .str.upper()
        short_able_df = short_able_df.set_index('asset')
        short_able = short_able_df['converRate'].to_dict()
        ret_df['short_able'] = ret_df.index.map(lambda x: short_able.get(x[1], 0))

        print('All ok')

        return ret_df
