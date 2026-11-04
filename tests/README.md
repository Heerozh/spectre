# Tests

```bash
uv run pytest -n 6                                  # with the CUDA build of torch, the default
uv run --no-group cuda --group cpu pytest -n auto   # with the CPU build, for machines without a GPU
```

With the CUDA build on Windows, keep the worker count around `-n 6`. Each process commits a lot of
memory for the CUDA DLLs, and `-n auto` on a many-core machine aborts with `WinError 1455` (page
file too small). A lower count can hang instead, because the `Pool` in `test_cpu_parallel_factor`
loads the DLLs again in every child process.

## Layout

The test folders mirror the `spectre` packages:

| Folder        | Tests for                                                     |
|---------------|---------------------------------------------------------------|
| `dataloader/` | `spectre.data`: csv/arrow loaders, `DataLoaderFastGetter`     |
| `factors/`    | `spectre.factors`: engine and all factors                     |
| `parallel/`   | `spectre.parallel`: group by, rolling, nan functions, ranking |
| `trading/`    | `spectre.trading`: portfolio, blotters, events, backtest      |
| `data/`       | test data, see `conftest.py` for its quirks                   |

Shared fixtures (loaders, engines, devices) are in `conftest.py`. Every test creates its own
engine, so tests are independent and can run in any order.

Expected values of factors are computed by the independent pandas implementations in
`factors/reference.py`, from the raw csv files, not by spectre itself. Use
`reference.expected()` for time-series factors and `reference.expected_xs()` for cross-section
factors, they apply the same delay as the engine.

## Markers

Some tests are skipped by default:

* `cuda`: runs only when CUDA is available. Tests using the `device` fixture run on both cpu
  and cuda.
* `network`: downloads data, run with `uv run pytest --run-network`.
* `external_data`: needs the quandl dataset (`wiki_prices.feather`, `WIKI_PRICES.zip`), set
  `SPECTRE_EXTERNAL_DATA` to the folder contains it, defaults to
  `../historical_data/us/prices/quandl` next to the repository.
* `slow`: takes minutes, run with `uv run --with alphalens-reloaded pytest --run-slow`. TODO: split
  and simplify them.
