# AGENTS.md

spectre is a open-source GPU-accelerated quantitative trading library built on PyTorch: a factor engine plus an
event-driven backtester. `README.md` is the API reference, and covers writing your own `DataLoader`
and factors.

## Tests

- Run `pytest -n auto`, see `tests/README.md`.
- Run one test with `pytest tests/factors/test_engine.py::test_run_raw`, or select by name with
  `pytest tests/factors -k "zscore or rank"`.
- There is no `pyproject.toml`, virtualenv or CI. With uv, this runs the tests in a throwaway
  environment. `spectre` does not need installing, pytest imports it from the repo root.

  ```bash
  uv run --no-project --python 3.13 --with-requirements requirements.txt --with-requirements requirements-test.txt pytest -n auto
  ```

- The library supports pandas 2.2 and 3. To test the old version, add
  `--with pandas==2.2.3 --with "numpy<2.3"` to that command. Without the numpy pin, pandas 2.2 emits
  deprecation warnings from newer numpy that have nothing to do with spectre.
- On Windows, PyPI's torch wheel is CPU-only. That environment skips every `cuda` test, so a passing
  run says nothing about GPU code paths.

## Lint and packaging

- There is no lint config. Lines are limited to 100 columns. `uvx flake8 --max-line-length=100
  tests/` is clean. `spectre/` has about 200 existing findings, mostly `__init__.py` re-exports.
- There is no build step. `setup.py` lists packages explicitly, so add any new subpackage there.
- Pandas version shims belong in `spectre/_compat.py`. Under pandas 3 Copy-on-Write, chained
  assignment silently does nothing, and `.to_numpy()` can return a read-only array that torch
  rejects.

## Architecture

- `spectre.parallel`: torch kernels with no knowledge of factors (`ParallelGroupBy`, `Rolling`,
  NaN-aware aggregations).
- `spectre.data`: loaders that return a long DataFrame indexed by `['date', 'asset']`.
- `spectre.factors`: the factor graph and `FactorEngine`. `CustomFactor` is the base for computed
  factors, with `CrossSectionFactor` and `FilterFactor` as its main variants. `ColumnDataFactor` and
  the other data factors sit directly on `BaseFactor`.
- `spectre.trading`: event loop, `CustomAlgorithm`, blotters and portfolio, on top of
  `FactorEngine`. The docstring in `spectre/trading/__init__.py` diagrams backtest and live flows.
- `spectre.plotting`: imported by `factors` and `trading` at module level. It imports them back
  lazily inside functions, so keep those imports local to avoid a cycle.

### How `FactorEngine.run` executes

1. `_run` wraps every factor where `should_delay()` is true in `.shift(1)`, then takes the largest
   `get_total_backwards_()` in the graph to decide how much history to load before `start`.
2. `_prepare_tensor` loads that range and builds the `asset` and `date` `ParallelGroupBy` objects.
   It skips the reload if the same range is already loaded with enough history.
3. `pre_compute_` walks the graph, binding the engine, counting references and moving data columns
   to tensors.
4. `CustomFactor.compute_` evaluates the graph recursively, keeping each result until its last
   consumer has read it (`_ref_count`). `_format_input` regroups each input to this factor's
   `groupby`, NaN-fills it by `mask`, and wraps it in `Rolling` if `win > 1`.
5. Results are reverted to row order, the filter drops rows, and the output is trimmed to `start`,
   or to the bar after `start` if anything was delayed.

### What `compute()` receives

The docstring of `CustomFactor.compute` has the full layouts. In short, `ParallelGroupBy` turns flat
rows into padded 2D tensors, using NaN or `False` for padding:

- `groupby = 'asset'` (the default) gives `[asset, bar]`.
- `groupby = 'date'` (`CrossSectionFactor`) gives `[date, asset]`. `groupby` can also be any
  DataFrame column, or several joined by commas.
- `win > 1` gives a `Rolling`, whose `.values` is `[asset, bar, win]`. Inputs that are not factors
  are passed through unchanged.
- All shapes include the history loaded before `start`, not only the requested range.
- A factor can return `[asset, bar, n]` for multiple outputs. Select one with `factor[i]` before
  adding it to the engine or rolling it.

### Things that are easy to get wrong

- **Factors overload `==`.** It returns an always-truthy `EqFactor` and leaves factors unhashable.
  `a == b` and `a in factor_list` are always true, and factors cannot go in sets or be dict keys.
  Compare with `is`.
- **Bars are not days.** `bar` is a position in each asset's own series, and `win` and `shift` count
  those positions. Columns of `[asset, bar]` line up across assets only with `align_by_time`, so do
  cross-sectional work through `groupby = 'date'`.
- **Auto delay.** Factors depending on high, low, close or volume are delayed one bar, because that
  data is only known after the close. `OHLCV.open` is not. `run` then drops the first bar. Control
  this with `delay_factor` (`True`, `False`, `'all'`) and `factor.set_delay()`.
- **Look-ahead is not prevented.** `compute()` receives every bar at once, including future ones.
  `engine.test_lookahead_bias` randomizes OHLCV in the second half of the range and checks that the
  first half is unchanged, but it cannot prove a factor clean: a one-bar leak on an auto-delayed
  factor cancels out against the delay and passes. The `run_factor` test helper calls it unless
  given `check_bias=False`.
- **Adjustments only happen inside `Rolling`, and only for a direct `ColumnDataFactor` input**, the
  only factor type with non-`None` `adjustments`. `MA(5, inputs=[close])` is re-adjusted per
  window, but `MA(5, inputs=[close.shift(1)])` and `MA(5, inputs=[close * 1])` are not. With
  `win = 1`, prices are raw. `Rolling.values` is also raw, so go through `agg()` or the aggregation
  methods.
- **`Rolling.agg` splits time into chunks** to limit memory. Its callback gets
  `[asset, chunk, win]`, must return `[asset, chunk]`, and cannot assume it sees every bar.
- **Filters are not masks.** `engine.set_filter` only drops output rows after everything is
  computed, so `close.rank()` still ranks all assets. Pass `mask=` to NaN-fill inputs instead. The
  string mask `'universe'` resolves to the engine's filter. `StaticAssets` is the exception: it
  pre-screens the loaded data.
- **Factors hold per-run state** (`_engine`, `_ref_count`, `_cache`) and instances are shared
  between runs and engines. `OHLCV.*` are class attributes whose `inputs` the engine rewrites from
  the loader on every run, `DatetimeDataFactor` is a per-attribute singleton, and class-level
  `inputs` such as `MarketReturn`'s `Returns()` are shared by all instances. Overrides of
  `pre_compute_` or `clean_up_` must call `super()`.
- **Tensors must be float or bool.** `ParallelGroupBy.split` raises on integers. The engine's
  default float type is `Global.float_type` in `spectre/config.py`.

### Data and backtesting

- Loaders implement `_load()` and `last_modified`. `DataLoader._format()` adds what the engine
  needs: `_time_cat_id`, and the `price_multi` and `vol_multi` columns, after moving `ex-dividend`
  and `split_ratio` up one row so they apply to bars before the ex-date. `test_load()` asserts the
  contract.
- A backtest calls `FactorEngine.run` once, for the whole range. `run_simulation_alg` then replaces
  `alg.run_engine` with a stub returning the rows up to the current bar, and replays those bars by
  firing `EveryBarData`, `MarketOpen` and `MarketClose`. In simulation only the sign of an event's
  offset matters: negative fires before the open or close, otherwise after.
- Blotter prices are raw. `SimulationBlotter.market_close` pushes dividends and splits to the
  portfolio, separately from the factor-side adjustments.
