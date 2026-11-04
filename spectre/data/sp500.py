"""
@author: Heerozh (Zhang Jianhao)
@copyright: Copyright 2026, Heerozh. All rights reserved.
@license: GPL-3.0-only
@email: heeroz@gmail.com
"""

import warnings

import pandas as pd

HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
# Wikipedia answers 403 to clients that don't identify themselves.
USER_AGENT = "spectre (https://github.com/Heerozh/spectre)"
# The table is edited by hand, there are cells like 'ALLE |'.
TICKER = r"([A-Z0-9.\-]+)"


def _load_changes() -> pd.DataFrame:
    """Changes of the index from Wikipedia, with the columns date, added and removed."""
    table = pd.read_html(
        CHANGES_URL,
        match="Effective Date",
        flavor="lxml",
        storage_options={"User-Agent": USER_AGENT},
    )[0]
    dates = table[("Effective Date", "Effective Date")]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates, format="mixed", errors="coerce"),
            "added": table[("Added", "Ticker")].str.extract(TICKER, expand=False),
            "removed": table[("Removed", "Ticker")].str.extract(TICKER, expand=False),
        }
    )


def sp500_constituents() -> pd.Series:
    """
    Members of the S&P 500 over time, since 1996. Some members are missing before 2001.

    The history is https://github.com/fja05680/sp500, which is updated by hand every couple of
    months. The changes after its last date come from Wikipedia. A renamed ticker is not a change
    of the index, so it is missing from those changes until the next update of the history.

    Tickers are the ones in use on that date, so they can't all be downloaded: many were delisted,
    and others now belong to a different company. For example `FB` is an ETF on Yahoo today, and
    `BBT` is another bank with a price history going back to 2000.

    :return: Series indexed by date, with a frozenset of tickers for each date the members changed.
        Use `.asof(date)` for the members on a date, and `.iloc[-1]` for the latest members.
    """
    tickers = pd.read_csv(HISTORY_URL, index_col="date", parse_dates=True)["tickers"]
    members = tickers.str.split(",").map(frozenset)
    members = members[members != members.shift()]
    last_date = members.index[-1]

    try:
        changes = _load_changes()
    except (OSError, ValueError, KeyError) as e:
        warnings.warn(
            f"Failed to get the changes of the S&P 500 from Wikipedia: {e!r}. "
            f"The members are only up to date until {last_date.date()}.",
            RuntimeWarning,
        )
        return members

    latest = members.iloc[-1]
    newer = {}
    for date, change in changes[changes["date"] > last_date].groupby("date"):
        latest = (latest - set(change["removed"].dropna())) | set(change["added"].dropna())
        newer[date] = latest
    if newer:
        members = pd.concat([members, pd.Series(newer)]).rename_axis("date")
    return members
