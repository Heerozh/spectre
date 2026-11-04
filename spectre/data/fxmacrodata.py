"""
FXMacroData daily FX reference-rate loader.
"""
import os

import pandas as pd
import requests

from .dataloader import DataLoader


FXMACRODATA_API_ROOT = "https://api.fxmacrodata.com/v1"
# The API returns at most 100 rows per request, newest first.
PAGE_LIMIT = 100
MAX_PAGES = 1000


def _split_pair(pair):
    pair = pair.upper().replace("/", "").replace("-", "").replace("_", "")
    if len(pair) != 6:
        raise ValueError("FX pair must be formatted like 'EURUSD' or 'EUR/USD'")
    return pair[:3], pair[3:]


def _clean_api_key(api_key):
    if not api_key:
        return None
    api_key = str(api_key).strip()
    if any(char.isspace() or not char.isprintable() for char in api_key):
        # Keep the key itself out of the message.
        raise ValueError("FXMacroData API key contains whitespace or control characters")
    return api_key or None


def _page_rows(payload):
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    detail = payload.get("detail") if isinstance(payload, dict) else None
    raise ValueError(
        "Unexpected FXMacroData response{}".format(": {}".format(detail) if detail else "")
    )


class FXMacroDataLoader(DataLoader):
    """Load FXMacroData daily reference rates as Spectre OHLCV data.

    FXMacroData publishes one official reference value per currency pair and
    date. The value is copied into open, high, low, and close with zero volume
    so it can be consumed by Spectre's normal daily-price data path.
    """

    def __init__(
        self,
        pair,
        start_date,
        end_date,
        api_key=None,
        api_root=FXMACRODATA_API_ROOT,
        session=None,
    ):
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        df = self.fetch(
            pair,
            start_date,
            end_date,
            api_key=api_key,
            api_root=api_root,
            session=session,
        )
        super().__init__("", ohlcv=("open", "high", "low", "close", "volume"), adjustments=None)
        self.df = self._format_fxmacrodata(df)
        self.test_load()

    @property
    def last_modified(self) -> float:
        return 1

    def _load(self):
        return self.df

    def _format_fxmacrodata(self, df):
        df = df.rename_axis(["date", "asset"]).reset_index()
        assets = sorted(pd.unique(df["asset"].astype(object)))
        asset_type = pd.api.types.CategoricalDtype(categories=assets, ordered=True)
        df["asset"] = df["asset"].astype(asset_type)
        df.set_index(["date", "asset"], inplace=True)
        if df.index.levels[0].tzinfo is None:
            df = df.tz_localize("UTC", level=0, copy=False)
        else:
            df = df.tz_convert("UTC", level=0, copy=False)
        df.sort_index(level=[0, 1], inplace=True)
        unique_date = df.index.get_level_values(0).unique()
        time_cat = dict(zip(unique_date, range(len(unique_date))))
        df[self.time_category] = df.index.get_level_values(0).map(time_cat)
        return df

    @classmethod
    def fetch(
        cls,
        pair,
        start_date,
        end_date,
        api_key=None,
        api_root=FXMACRODATA_API_ROOT,
        session=None,
    ):
        base, quote = _split_pair(pair)
        api_key = _clean_api_key(api_key or os.environ.get("FXMACRODATA_API_KEY"))
        headers = {"X-API-Key": api_key} if api_key else {}
        client = session or requests
        url = "{}/forex/{}/{}".format(api_root.rstrip("/"), base, quote)
        rows = []
        offset = 0
        for _ in range(MAX_PAGES):
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "limit": PAGE_LIMIT,
                "offset": offset,
            }
            # Redirects are not followed so the key header is never forwarded
            # to another host.
            response = client.get(
                url, params=params, headers=headers, timeout=30, allow_redirects=False
            )
            if 300 <= response.status_code < 400:
                raise requests.HTTPError(
                    "FXMacroData returned an unexpected redirect (HTTP {})".format(
                        response.status_code
                    ),
                    response=response,
                )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:
                raise ValueError("FXMacroData returned a non-JSON response") from None
            page = _page_rows(payload)
            rows.extend(page)
            pagination = payload.get("pagination") or {}
            if not page or not pagination.get("has_more"):
                break
            next_offset = pagination.get("next_offset")
            if next_offset is None:
                next_offset = offset + len(page)
            if next_offset <= offset:
                break
            offset = next_offset

        records = []
        asset = "{}/{}".format(base, quote)
        for row in rows:
            if not isinstance(row, dict) or row.get("val") is None or not row.get("date"):
                continue
            try:
                value = float(row["val"])
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "date": pd.to_datetime(row["date"]),
                    "asset": asset,
                    "open": value,
                    "high": value,
                    "low": value,
                    "close": value,
                    "volume": 0.0,
                }
            )

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"]),
            )
        df["asset"] = df["asset"].astype(object)
        return df.set_index(["date", "asset"]).sort_index()
