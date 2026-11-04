"""
Tests need data outside of this repository or the internet, they are skipped by default.
"""

import pytest
import torch
from numpy.testing import assert_almost_equal

import spectre
from tests.conftest import external_data


@pytest.mark.network
def test_yahoo(tmp_path):
    spectre.data.YahooDownloader.ingest(
        "2011", str(tmp_path) + "/", ["IBM", "AAPL"], skip_exists=False
    )
    loader = spectre.data.ArrowLoader(str(tmp_path / "yahoo.feather"))
    df = loader._load()
    assert list(df.index.levels[1]) == ["AAPL", "IBM"]


@pytest.mark.slow
@pytest.mark.external_data
def test_quandl_loader(tmp_path, monkeypatch):
    zip_file = external_data("WIKI_PRICES.zip")
    # ingest to a temporary file, don't touch the shared data used by other tests
    feather = str(tmp_path / "wiki_prices.feather")
    spectre.data.ArrowLoader.ingest(spectre.data.QuandlLoader(str(zip_file)), feather)
    loader = spectre.data.ArrowLoader(feather)

    monkeypatch.setattr(spectre.parallel.Rolling, "_split_multi", 80)
    engine = spectre.factors.FactorEngine(loader)
    engine.add(spectre.factors.MA(100), "ma")
    if torch.cuda.is_available():
        engine.to_cuda()
    df = engine.run("2014-01-02", "2014-01-02", delay_factor=False)
    # expected result comes from zipline
    assert_almost_equal(
        df.head().values.T, [[51.388700, 49.194407, 599.280580, 28.336585, 12.7058]], decimal=4
    )
    assert_almost_equal(
        df.tail().values.T, [[86.087988, 3.602880, 7.364000, 31.428209, 27.605950]], decimal=4
    )

    # test last line bug
    engine.run("2016-12-15", "2017-01-02")
    df = engine._dataframe.loc[(slice("2016-12-15", "2017-12-15"), "STJ"), :]
    assert df.price_multi.values[-1] == 1
