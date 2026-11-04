"""
Tests need the quandl dataset outside of this repository, they are skipped by default.
See `EXTERNAL_DATA_DIR` of tests/conftest.py.
"""

import gc

import pytest
import torch
from numpy.testing import assert_almost_equal

import spectre
from tests.conftest import external_data


@pytest.fixture
def quandl_loader():
    return spectre.data.ArrowLoader(str(external_data("wiki_prices.feather")))


@pytest.mark.cuda
@pytest.mark.external_data
def test_memory_leak(quandl_loader):
    engine = spectre.factors.FactorEngine(quandl_loader)
    engine.to_cuda()
    universe = spectre.factors.AverageDollarVolume(win=252).top(500)
    engine.set_filter(universe)
    engine.get_price_matrix("2014-01-02", "2014-01-12")
    del engine, universe
    spectre.parallel.DeviceConstant.clean()

    gc.collect(2)
    gc.collect(2)
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == 0


@pytest.mark.slow
@pytest.mark.external_data
def test_full_run_same_as_alphalens(quandl_loader):
    al = pytest.importorskip("alphalens")
    engine = spectre.factors.FactorEngine(quandl_loader)
    if torch.cuda.is_available():
        engine.to_cuda()
    universe = spectre.factors.AverageDollarVolume(win=120).top(500)
    engine.set_filter(universe)
    f = spectre.factors.MA(5) - spectre.factors.MA(10) - spectre.factors.MA(30)
    engine.add(f.rank(mask=universe).zscore(), "ma_cross")
    df_prices = engine.get_price_matrix("2014-01-02", "2017-01-18")
    factor_data, mean_return = engine.full_run(
        "2014-01-02", "2017-01-19", periods=(10,), filter_zscore=None, preview=False
    )

    al_clean_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_data[("ma_cross", "factor")],
        prices=df_prices,
        periods=[10],
        filter_zscore=None,
    )
    al_mean, al_std = al.performance.mean_return_by_quantile(al_clean_data)
    assert_almost_equal(
        mean_return[("ma_cross", "10D", "mean")].values, al_mean["10D"].values, decimal=5
    )
    assert_almost_equal(
        mean_return[("ma_cross", "10D", "sem")].values, al_std["10D"].values, decimal=5
    )

    # check last line bug, nanlast function
    stj_10d_return = factor_data.loc[(slice("2016-12-15", "2017-01-03"), "STJ"), ("Returns", "10D")]
    expected = [0.003880821, 0.01881325, 0.01290882, 0.01762784, 0.0194248, 0.01405275,
                0.01240134, 0.00835931, 0.01265514, 0.01240134, 0.00785625, 0.00161111]  # fmt: skip
    assert_almost_equal(stj_10d_return.values, expected)

    # test zscore filter
    factor_data, mean_return = engine.full_run(
        "2014-01-02", "2017-01-19", periods=(10,), filter_zscore=0.5, preview=False
    )
    al_clean_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_data[("ma_cross", "factor")],
        prices=df_prices,
        periods=[10],
        filter_zscore=0.5,
        max_loss=1,
    )
    al_mean, al_std = al.performance.mean_return_by_quantile(al_clean_data)
    # alphalens filter zscore only uses returned data, spectre includes backward data, so
    # only compares with lower precision.
    assert_almost_equal(
        mean_return[("ma_cross", "10D", "mean")].values, al_mean["10D"].values, decimal=3
    )
    assert_almost_equal(
        mean_return[("ma_cross", "10D", "sem")].values, al_std["10D"].values, decimal=3
    )
