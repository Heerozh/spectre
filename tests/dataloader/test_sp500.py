"""
sp500_constituents without the network, `test_external.py` has the real one.
"""

from urllib.error import URLError

import pandas as pd
import pytest

from spectre.data import sp500, sp500_constituents


def wikipedia_table(rows):
    columns = [
        ("Effective Date", "Effective Date"),
        ("Added", "Ticker"),
        ("Added", "Security"),
        ("Removed", "Ticker"),
        ("Removed", "Security"),
        ("Reason", "Reason"),
        ("Refs", "Refs"),
    ]
    full_rows = [(date, added, added, removed, removed, "", "") for date, added, removed in rows]
    return pd.DataFrame(full_rows, columns=pd.MultiIndex.from_tuples(columns))


@pytest.fixture
def history(tmp_path, monkeypatch):
    csv = tmp_path / "history.csv"
    csv.write_text(
        "date,tickers\n"
        '2020-01-02,"AAA,BBB,CCC"\n'
        '2020-01-03,"AAA,BBB,CCC"\n'
        '2020-02-03,"AAA,BBB,DDD"\n'
    )
    monkeypatch.setattr(sp500, "HISTORY_URL", str(csv))


def test_constituents_applies_newer_changes(history, monkeypatch):
    table = wikipedia_table(
        [
            ("October 30, 2026", "OKTA", "BBB"),
            ("October 6, 2026", None, "DDD"),
            ("October 1, 2026", "EEE |", None),
            ("January 5, 2020", "OLD", "AAA"),  # already part of the history
        ]
    )
    monkeypatch.setattr(pd, "read_html", lambda *args, **kwargs: [table])
    members = sp500_constituents()
    assert members.index.name == "date"
    assert list(members.index) == list(
        pd.to_datetime(["2020-01-02", "2020-02-03", "2026-10-01", "2026-10-06", "2026-10-30"])
    )
    assert members.asof("2020-01-20") == {"AAA", "BBB", "CCC"}
    assert members.asof("2020-02-03") == {"AAA", "BBB", "DDD"}
    assert members.asof("2026-10-02") == {"AAA", "BBB", "DDD", "EEE"}
    assert members.iloc[-1] == {"AAA", "EEE", "OKTA"}


def test_constituents_without_newer_changes(history, monkeypatch):
    table = wikipedia_table([("January 5, 2020", "OLD", "AAA")])
    monkeypatch.setattr(pd, "read_html", lambda *args, **kwargs: [table])
    members = sp500_constituents()
    assert isinstance(members.index, pd.DatetimeIndex)
    assert members.iloc[-1] == {"AAA", "BBB", "DDD"}


@pytest.mark.parametrize(
    "failure", [URLError("offline"), ValueError("No tables found"), KeyError("Added")]
)
def test_constituents_warns_if_wikipedia_fails(history, monkeypatch, failure):
    def read_html(*args, **kwargs):
        raise failure

    monkeypatch.setattr(pd, "read_html", read_html)
    with pytest.warns(RuntimeWarning, match="only up to date until 2020-02-03"):
        members = sp500_constituents()
    assert members.iloc[-1] == {"AAA", "BBB", "DDD"}
