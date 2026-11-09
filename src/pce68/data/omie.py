"""OMIE day-ahead marginal prices (marginalpdbc files).

Every day OMIE publishes marginalpdbc_YYYYMMDD.1:

    MARGINALPDBC;
    2024;01;01;1;63.33;63.33;
    ...
    *

Columns: year; month; day; period; Portugal price; Spain price (€/MWh).
Periods count from local midnight, starting at 1. Up to 30/09/2025 there are
24 hourly periods (23 or 25 on daylight-saving days); from 01/10/2025 the
day-ahead market clears in quarter hours: 96 periods (92 or 100).

UTC time is computed by adding durations to local midnight, which is never
ambiguous, so daylight-saving changes come out right without special handling.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TZ = "Europe/Madrid"
URL_BASE = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename="
URL = URL_BASE + "marginalpdbc_{:%Y%m%d}.1"
# OMIE only serves recent files one day at a time; closed years are bundled
# into a yearly zip that holds the same marginalpdbc_YYYYMMDD.1 files.
URL_YEARLY = URL_BASE + "marginalpdbc_{}.zip"


def file_name(day: date) -> str:
    return f"marginalpdbc_{day:%Y%m%d}.1"


def parse(text: str) -> pd.DataFrame:
    """Turn a marginalpdbc file into rows of timestamp_utc, price_es, price_pt, resolution_min."""
    lines = [ln for ln in text.splitlines() if ln[:1].isdigit()]
    if not lines:
        raise ValueError("marginalpdbc file without data")
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=";",
        header=None,
        usecols=range(6),
        names=["year", "month", "day", "period", "price_pt", "price_es"],
    )
    if df[["year", "month", "day"]].nunique().max() > 1:
        raise ValueError("a marginalpdbc file must hold a single day")

    y, m, d = df.loc[0, ["year", "month", "day"]]
    day = pd.Timestamp(year=int(y), month=int(m), day=int(d))
    start = day.tz_localize(TZ)
    end = (day + pd.Timedelta(days=1)).tz_localize(TZ)  # next local midnight: 23, 24 or 25 h days
    day_hours = (end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1)
    n = len(df)
    resolution = 60.0 * day_hours / n
    if resolution not in (15.0, 60.0):
        raise ValueError(f"{y}-{m:02d}-{d:02d}: {n} periods do not fit a {day_hours:g} h day")

    t0 = start.tz_convert("UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": t0 + pd.to_timedelta((df["period"] - 1) * resolution, unit="min"),
            "price_es": df["price_es"].astype(float),
            "price_pt": df["price_pt"].astype(float),
            "resolution_min": int(resolution),
        }
    )


def _download_yearly(s, year: int, dest: Path) -> int:
    """Fetch the yearly zip and extract its marginalpdbc files into `dest`. Returns how many."""
    r = s.get(URL_YEARLY.format(year), timeout=120)
    r.raise_for_status()
    n = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            base = Path(name).name
            if base.startswith("marginalpdbc_") and base.endswith(".1"):
                (dest / base).write_bytes(z.read(name))
                n += 1
    return n


def download(days: list[date], dest: Path, pause_s: float = 0.2) -> list[date]:
    """Download the files missing from `dest`. Returns the days that failed.

    Tries the daily file first; if OMIE no longer serves it on its own (older
    years), tries that year's zip once.
    """
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    years_tried: set[int] = set()
    with requests.Session() as s:
        for day in days:
            path = dest / file_name(day)
            if path.exists() and path.stat().st_size > 0:
                continue
            try:
                r = s.get(URL.format(day), timeout=30)
                r.raise_for_status()
                parse(r.text)  # never save an error page under a valid file name
                path.write_text(r.text, encoding="latin-1")
                time.sleep(pause_s)
                continue
            except Exception as e:  # noqa: BLE001 - report it and carry on with the rest
                error = e
            if day.year not in years_tried:
                years_tried.add(day.year)
                try:
                    n = _download_yearly(s, day.year, dest)
                    log.info("%d: yearly zip with %d days", day.year, n)
                except Exception as e:  # noqa: BLE001
                    log.warning("%d: no yearly zip either (%s)", day.year, e)
                if path.exists() and path.stat().st_size > 0:
                    continue
            log.warning("%s: %s", day, error)
            failed.append(day)
    return failed


def consolidate(source: Path) -> pd.DataFrame:
    """Join every marginalpdbc file in `source` into one sorted, de-duplicated series."""
    frames = [parse(f.read_text(encoding="latin-1")) for f in sorted(source.glob("marginalpdbc_*.1"))]
    if not frames:
        raise FileNotFoundError(f"no marginalpdbc files in {source}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc", ignore_index=True)
