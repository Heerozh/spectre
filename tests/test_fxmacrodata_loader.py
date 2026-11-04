import unittest

import pandas as pd
import requests

from spectre.data import FXMacroDataLoader


class TestFXMacroDataLoader(unittest.TestCase):

    def test_fetch_formats_fxmacrodata_rows(self):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "data": [
                        {"date": "2026-01-02", "val": 1.2},
                        {"date": "2026-01-01", "val": 1.1},
                    ]
                }

        class MockSession:
            def __init__(self):
                self.calls = []

            def get(self, url, params, headers, timeout, **kwargs):
                self.calls.append((url, params, headers, timeout))
                return MockResponse()

        session = MockSession()
        df = FXMacroDataLoader.fetch(
            "EUR/USD",
            "2026-01-01",
            "2026-01-02",
            api_key="test-key",
            api_root="https://example.test/api/v1",
            session=session,
        )

        self.assertEqual(
            session.calls[0][0],
            "https://example.test/api/v1/forex/EUR/USD",
        )
        self.assertEqual(session.calls[0][2], {"X-API-Key": "test-key"})
        self.assertEqual(df.index.names, ["date", "asset"])
        self.assertEqual(df.index.get_level_values("asset").unique()[0], "EUR/USD")
        self.assertEqual(list(df["close"]), [1.1, 1.2])

    def test_fetch_follows_pagination(self):
        pages = {
            0: {
                "data": [{"date": "2026-01-02", "val": 1.2}],
                "pagination": {"has_more": True, "next_offset": 1},
            },
            1: {
                "data": [{"date": "2026-01-01", "val": 1.1}],
                "pagination": {"has_more": False, "next_offset": None},
            },
        }

        class MockResponse:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self.payload

        class MockSession:
            def __init__(self):
                self.offsets = []

            def get(self, url, params, headers, timeout, **kwargs):
                self.offsets.append((params["limit"], params["offset"]))
                return MockResponse(pages[params["offset"]])

        session = MockSession()
        df = FXMacroDataLoader.fetch(
            "EURUSD", "2026-01-01", "2026-01-02", session=session
        )

        self.assertEqual(session.offsets, [(100, 0), (100, 1)])
        self.assertEqual(list(df["close"]), [1.1, 1.2])

    def test_loader_returns_spectre_formatted_data(self):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"date": "2026-01-01", "val": 1.1}]}

        class MockSession:
            def get(self, url, params, headers, timeout, **kwargs):
                return MockResponse()

        loader = FXMacroDataLoader(
            "EURUSD",
            "2026-01-01",
            "2026-01-01",
            session=MockSession(),
        )
        df = loader.load()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.index.names, ["date", "asset"])
        self.assertIn("volume", df.columns)


    def test_fetch_does_not_follow_redirects(self):
        class MockResponse:
            status_code = 302

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class MockSession:
            def __init__(self):
                self.kwargs = None

            def get(self, url, params, headers, timeout, **kwargs):
                self.kwargs = kwargs
                return MockResponse()

        session = MockSession()
        with self.assertRaises(requests.HTTPError):
            FXMacroDataLoader.fetch(
                "EURUSD", "2026-01-01", "2026-01-02", api_key="test-key", session=session
            )
        self.assertIs(session.kwargs.get("allow_redirects"), False)

    def test_invalid_api_key_error_does_not_include_key(self):
        with self.assertRaises(ValueError) as ctx:
            FXMacroDataLoader.fetch(
                "EURUSD", "2026-01-01", "2026-01-02", api_key="test\nkey", session=object()
            )
        self.assertNotIn("test", str(ctx.exception))

    def test_error_body_raises_clean_error(self):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"detail": "Invalid API key"}

        class MockSession:
            def get(self, url, params, headers, timeout, **kwargs):
                return MockResponse()

        with self.assertRaisesRegex(ValueError, "Invalid API key"):
            FXMacroDataLoader.fetch(
                "EURUSD", "2026-01-01", "2026-01-02", session=MockSession()
            )


if __name__ == "__main__":
    unittest.main()
