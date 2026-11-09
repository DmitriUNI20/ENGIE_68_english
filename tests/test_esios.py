from datetime import date

import pandas as pd
import pytest

from pce68.data import esios


class FakeSession:
    """Answers like ESIOS: one indicator, several zones, hourly values."""

    def __init__(self, values_per_call, catalogue=None):
        self.calls = []
        self.values_per_call = values_per_call
        self.catalogue = catalogue or []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        body = (
            {"indicators": self.catalogue}
            if url.endswith("/indicators")
            else {"indicator": {"name": "Demanda real", "values": self.values_per_call}}
        )

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return body

        return R()


def hourly(day: str, hours: int, zone: str, value: float) -> list[dict]:
    t0 = pd.Timestamp(day, tz="UTC")
    return [
        {"value": value, "datetime_utc": (t0 + pd.Timedelta(hours=h)).isoformat(), "geo_name": zone}
        for h in range(hours)
    ]


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("ESIOS_TOKEN", "x" * 64)


def test_hourly_values_spread_over_four_quarters():
    s = esios.to_quarter_hours(pd.DataFrame(hourly("2026-01-01", 2, "Península", 100.0)))
    assert len(s) == 8 and (s == 100.0).all()


def test_five_minute_values_are_averaged():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    values = pd.DataFrame(
        {"value": [1, 2, 3, 4, 5, 6], "datetime_utc": [t0 + pd.Timedelta(minutes=5 * i) for i in range(6)]}
    )
    assert esios.to_quarter_hours(values).tolist() == [2.0, 5.0]


def test_only_the_peninsula_is_kept():
    values = pd.DataFrame(
        hourly("2026-01-01", 1, "Península", 100.0) + hourly("2026-01-01", 1, "Baleares", 5.0)
    )
    assert esios.peninsula(values)["value"].tolist() == [100.0]


def test_indicator_reads_name_values_and_sends_the_token_header():
    session = FakeSession(hourly("2026-01-01", 24, "Península", 30000.0))
    name, series = esios.indicator(session, 1293, date(2026, 1, 1), date(2026, 1, 1))
    assert name == "Demanda real"
    assert len(series) == 96 and series.iloc[0] == 30000.0
    url, params, headers = session.calls[0]
    assert url.endswith("/indicators/1293") and params["start_date"].startswith("2026-01-01")
    assert headers["x-api-key"] == "x" * 64


def test_long_ranges_are_fetched_in_chunks():
    session = FakeSession(hourly("2026-01-01", 1, "Península", 1.0))
    esios.indicator(session, 1293, date(2026, 1, 1), date(2026, 3, 31))
    assert len(session.calls) == 3  # 90 days in chunks of 30


def test_find_ignores_case_and_accents():
    catalogue = [{"id": 551, "name": "Generación T.Real eólica"}, {"id": 1293, "name": "Demanda real"}]
    hits = esios.find(FakeSession([], catalogue), "EOLICA")
    assert hits["id"].tolist() == [551]


def test_missing_token_stops_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ESIOS_TOKEN")
    with pytest.raises(SystemExit, match="ESIOS_TOKEN"):
        esios.token()


def test_resolve_takes_ids_and_looks_up_names(monkeypatch):
    import pytest

    from pce68.data import esios

    catalogue = {
        "indicators": [
            {"id": 546, "name": "Generación T.Real hidráulica"},
            {"id": 1, "name": "Generación T.Real hidráulica bombeo"},
            {"id": 549, "name": "Generación T.Real nuclear"},
        ]
    }
    monkeypatch.setattr(esios, "_get", lambda session, path, params=None: catalogue)
    assert esios.resolve(None, 1293) == 1293 and esios.resolve(None, "551") == 551
    assert esios.resolve(None, "generacion t.real HIDRAULICA") == 546  # exact match wins over the longer name
    assert esios.resolve(None, "nuclear") == 549  # the only one containing it
    with pytest.raises(ValueError, match="3 indicators"):
        esios.resolve(None, "T.Real")  # ambiguous: listed so the user can pick an id
