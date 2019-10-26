from typing import Union, Optional, Iterable
from .factor import BaseFactor, IndexFactor
import pandas as pd
import os

"""
df1 = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], columns=('a','b','c'))
df2 = pd.DataFrame([[1,20,30],[4,50,60],[7,80,90]], columns=('a','b','c'))
df1 = df1.set_index('a')
df2 = df2.set_index('a')
Out[66]: 
   b  c
a      
1  2  3
4  5  6
7  8  9
Out[67]: 
    b   c
a        
1  20  30
4  50  60
7  80  90

df = pd.concat({'df1': df1, 'df2':df2})
Out[87]: 
        b   c
    a        
df1 1   2   3
    4   5   6
    7   8   9
df2 1  20  30
    4  50  60
    7  80  90
    
df['b'].unstack(level=[0])   
Out[88]: 
   df1  df2
a          
1    2   20
4    5   50
7    8   80
 
"""


class DataLoader:
    def load(self, start, end) -> pd.DataFrame:
        """
        Get the data, and TODO: adjust for Dividends/Splits (anchor at `end`)
        If for back-testing, start and end should be same.
        """
        raise NotImplementedError("abstractmethod")


class CsvDirDataLoader(DataLoader):
    def __init__(self, path: str, split_by_year=False, **read_csv):
        self._csv_dir = path
        # dividends: self._div_dir = dividends_path #dividends_path='',
        self._split_by_year = split_by_year
        self._read_csv = read_csv

    def _load_split_by_year(self, start, end):
        years = set(pd.date_range(start, end).year)
        dfs = {}
        for entry in os.scandir(self._csv_dir):
            if entry.name.endswith(".csv") and entry.is_file():
                # 'spy_2011.csv'
                year = entry.name[-8:-4]
                name = entry.name[:-9]
                if int(year) not in years:
                    continue
                if name in dfs:
                    continue
                df = pd.DataFrame()
                for year in years:
                    try:
                        df_year = pd.read_csv(self._csv_dir + '{}_{}.csv'.format(name, year),
                                              **self._read_csv)
                        df = pd.concat([df_year, df])
                    except OSError:
                        pass
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep='last')]
                df = df.loc[start:end]
                dfs[name] = df
        return pd.concat(dfs)

    def _load(self, start, end):
        dfs = {}
        for entry in os.scandir(self._csv_dir):
            if entry.name.endswith(".csv") and entry.is_file():
                df = pd.read_csv(self._csv_dir + entry.name, **self._read_csv)
                df = df.loc[start:end]
                dfs[entry.name[:-4]] = df
        return pd.concat(dfs)

    def load(self, start, end):
        if self._split_by_year:
            return self._load_split_by_year(start, end)
        else:
            return self._load(start, end)


class FactorEngine:
    """
    Engine for compute factors, used for back-testing and alpha-research both.
    Engine do not process dividends, please use adjustment data.
    Dividends/Split only plays a role in the back-test, in order to simulate the real situation.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataframe = None
        self._factors = {}

    def add(self, factor: Union[Iterable[BaseFactor], BaseFactor],
            name: Union[Iterable[str], str]) -> None:
        if isinstance(factor, Iterable):
            for i, fct in enumerate(factor):
                self.add(fct, name and name[i] or None)
        else:
            if name in self._factors:
                raise KeyError('A factor with the name {} already exists.'
                               'please specify a new name by engine.add(factor, new_name)'
                               .format(name))
            self._factors[name] = factor

    def run(self, start: Optional[any] = None,
            end: Optional[any] = None) -> pd.DataFrame:

        self._dataframe = self._loader.load(start, end)

        # Calculate data that requires backward in tree
        for f in self._factors.values():
            f._clean()
        for f in self._factors.values():
            f._update_backward()

        # Compute factors
        for c, f in self._factors.items():
            f.pre_compute(start, end)
            f._compute(df[c])

        return df
