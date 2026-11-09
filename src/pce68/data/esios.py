"""REE's ESIOS API: demand and generation by technology (the market fundamentals).

    GET https://api.esios.ree.es/indicators/{id}?start_date=...&end_date=...
    header  x-api-key: <token>

The answer is {"indicator": {"name": ..., "values": [{"value", "datetime_utc",
"geo_id", "geo_name", ...}, ...]}}. Some indicators come for several zones
(peninsula, Balearic and Canary islands...); only the peninsula is kept, as the
day-ahead price is set there. Values arrive at the indicator's own resolution
(5 min, 15 min or hourly) and are averaged to quarter hours, the resolution of
the day-ahead market since 01/10/2025. Hourly values are spread over their four
quarters.

The token is read from the ESIOS_TOKEN environment variable and is never
written to any file of the project. It is free; ask for it at
consultasios@ree.es.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd

URL = "https://api.esios.ree.es"
HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
}
CHUNK_DAYS = 30  # ESIOS times out on long ranges for indicators with many zones


def token() -> str:
    value = os.environ.get("ESIOS_TOKEN", "").strip()
    if not value:
        raise SystemExit(
            'ESIOS_TOKEN is not set. On Windows: setx ESIOS_TOKEN "<your token>", then reopen the terminal.'
        )
    return value


def _get(session, path: str, params: dict | None = None) -> dict:
    r = session.get(f"{URL}/{path}", params=params, headers={**HEADERS, "x-api-key": token()}, timeout=60)
    r.raise_for_status()
    return r.json()


def find(session, text: str) -> pd.DataFrame:
    """Indicators of the ESIOS catalogue whose name contains `text` (case and accents ignored)."""
    catalogue = pd.DataFrame(_get(session, "indicators").get("indicators", []))
    if catalogue.empty:
        return catalogue

    def fold(s: pd.Series) -> pd.Series:
        return s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.lower()

    hit = fold(catalogue["name"].fillna("")).str.contains(fold(pd.Series([text]))[0], regex=False)
    return catalogue.loc[hit, ["id", "name"]].sort_values("id").reset_index(drop=True)


def peninsula(values: pd.DataFrame) -> pd.DataFrame:
    """Keep the peninsular zone when an indicator comes for several."""
    if "geo_name" not in values or values["geo_name"].nunique() <= 1:
        return values
    pen = values["geo_name"].str.contains("pen", case=False, na=False)
    if not pen.any():
        raise ValueError(f"no peninsular zone among {sorted(values['geo_name'].unique())}")
    return values[pen]


def to_quarter_hours(values: pd.DataFrame) -> pd.Series:
    """Indicator values at their own resolution -> one mean value per quarter hour (UTC)."""
    v = values.assign(t=pd.to_datetime(values["datetime_utc"], utc=True)).sort_values("t")
    s = v.groupby("t")["value"].mean()
    step = s.index.to_series().diff().median() if len(s) > 1 else pd.Timedelta(minutes=15)
    if step >= pd.Timedelta(minutes=60):
        # Hourly: the same value for each of the hour's four quarters.
        s = s.reindex(pd.date_range(s.index[0], s.index[-1] + pd.Timedelta(minutes=45), freq="15min")).ffill(
            limit=3
        )
    return s.resample("15min").mean()


def indicator(session, indicator_id: int, start: date, end: date) -> tuple[str, pd.Series]:
    """(official name, quarter-hourly peninsular values) of an indicator between two days."""
    name, chunks = "", []
    day = start
    while day <= end:
        last = min(day + timedelta(days=CHUNK_DAYS - 1), end)
        data = _get(
            session,
            f"indicators/{indicator_id}",
            {"start_date": f"{day:%Y-%m-%d}T00:00", "end_date": f"{last:%Y-%m-%d}T23:59"},
        ).get("indicator", {})
        name = data.get("name", name)
        chunks.append(pd.DataFrame(data.get("values", [])))
        day = last + timedelta(days=1)
    values = pd.concat(chunks, ignore_index=True)
    if values.empty:
        raise ValueError(f"indicator {indicator_id} ({name}) returned no values")
    return name, to_quarter_hours(peninsula(values))
