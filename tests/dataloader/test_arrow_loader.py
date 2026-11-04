import pandas as pd
import pytest

import spectre


@pytest.fixture
def ingested(tmp_path, adjusted_daily_loader):
    path = str(tmp_path / "prices.feather")
    spectre.data.ArrowLoader.ingest(adjusted_daily_loader, path)
    return path


def test_ingest_and_load(ingested, adjusted_daily_loader):
    loader = spectre.data.ArrowLoader(ingested)
    assert tuple(loader.ohlcv) == adjusted_daily_loader.ohlcv
    assert tuple(loader.adjustments) == adjusted_daily_loader.adjustments
    pd.testing.assert_frame_equal(loader.test_load(), adjusted_daily_loader.load())


def test_ingest_without_adjustments(tmp_path, daily_loader):
    path = str(tmp_path / "prices.feather")
    spectre.data.ArrowLoader.ingest(daily_loader, path)
    loader = spectre.data.ArrowLoader(path)
    assert tuple(loader.ohlcv) == daily_loader.ohlcv
    assert loader.adjustments is None
    pd.testing.assert_frame_equal(loader.load(), daily_loader.load())


def test_ingest_skipped_when_source_unchanged(ingested, adjusted_daily_loader):
    with pytest.warns(RuntimeWarning, match="no ingestion required"):
        spectre.data.ArrowLoader.ingest(adjusted_daily_loader, ingested)
    spectre.data.ArrowLoader.ingest(adjusted_daily_loader, ingested, force=True)


def test_meta_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        spectre.data.ArrowLoader(str(tmp_path / "not_exists.feather"))


def test_filter(ingested):
    loader = spectre.data.ArrowLoader(ingested)
    loader.filter(lambda df: df.loc[(slice(None), ["AAPL"]), :])
    df = loader.load()
    assert set(df.index.get_level_values(1)) == {"AAPL"}


def test_engine_with_arrow_loader(ingested, adjusted_daily_loader):
    def run(loader):
        engine = spectre.factors.FactorEngine(loader)
        engine.add(spectre.factors.MA(5), "ma5")
        return engine.run("2019-01-01", "2019-01-15")

    pd.testing.assert_frame_equal(
        run(spectre.data.ArrowLoader(ingested)), run(adjusted_daily_loader)
    )
