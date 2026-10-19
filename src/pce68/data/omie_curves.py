"""OMIE aggregated supply and demand curves (curva_pbc files).

Alongside the cleared price, OMIE publishes the curves the price came from:
one file per day holding, for every clearing period, the bids and offers
aggregated into price steps. They are public with roughly a ninety-day delay
and the archive is a rolling six years, which is exactly the span this project
covers -- what is not downloaded now is lost.

    curva_pbc_YYYYMMDD.1

The reader finds its columns by name rather than position, because the layout
differs between the aggregated files and the monthly curva_pbc_uof ones, and
because from 01/10/2025 the period field counts quarter hours instead of hours,
matching the change already handled in `pce68.data.omie`.

The exact layout has not been checked against a live file from this project, so
the first run on real data must be validated: re-clear the untouched historical
curves with `pce68.market.curves.clear` and compare against the published
marginalpdbc price. Gottfried and Muller report a mean error of 0.05 EUR/MWh in
2024 from ignoring block bids and market coupling; an error of that order means
the reader works, an error of several euros means it does not.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pce68.market.curves import Curve

log = logging.getLogger(__name__)

URL = "https://www.omie.es/es/file-download?parents%5B0%5D=curva_pbc&filename=curva_pbc_{:%Y%m%d}.1"
ENCODING = "latin-1"

# Fields we need, and the fragments that identify them whatever the layout.
WANTED = {
    "period": ("periodo", "hora"),
    "kind": ("tipo",),
    "energy": ("energ",),
    "price": ("precio",),
    "state": ("ofertada", "casada"),
}


def file_name(day: date) -> str:
    return f"curva_pbc_{day:%Y%m%d}.1"


def _fold(text: str) -> str:
    """Lowercase without accents or padding, so field names compare reliably."""
    stripped = unicodedata.normalize("NFKD", text)
    return "".join(c for c in stripped if not unicodedata.combining(c)).lower().strip()


def _header_row(lines: list[str]) -> int:
    """Index of the header line: the first one naming both a price and an energy."""
    for i, line in enumerate(lines[:20]):
        folded = _fold(line)
        if "precio" in folded and "energ" in folded:
            return i
    raise ValueError("no header found; check the OMIE file layout")


def _columns(header: list[str]) -> dict[str, int]:
    """Position of each wanted field within the header."""
    found: dict[str, int] = {}
    for i, field in enumerate(header):
        folded = _fold(field)
        if not folded:
            continue
        for name, fragments in WANTED.items():
            if name not in found and any(f in folded for f in fragments):
                found[name] = i
                break
    missing = {"period", "kind", "energy", "price"} - found.keys()
    if missing:
        raise ValueError(f"OMIE header is missing {sorted(missing)}")
    return found


def _number(text: str) -> float:
    """Spanish number (dot thousands, comma decimals) to float."""
    cleaned = text.strip().replace(".", "").replace(",", ".")
    return np.nan if cleaned in ("", "-") else float(cleaned)


def parse(text: str, offered_only: bool = True) -> pd.DataFrame:
    """Turn a curva_pbc file into rows of period, kind, energy_mwh, price_eur_mwh.

    `kind` is "C" for a bid (compra) and "V" for an offer (venta). When the file
    marks offered and matched points apart, only the offered ones are kept:
    those are the full curve, and the matched subset would clear at itself.
    """
    lines = text.splitlines()
    columns = _columns(lines[_header_row(lines)].split(";"))
    width = max(columns.values()) + 1

    rows: list[tuple[int, str, float, float]] = []
    skipped = 0
    for line in lines[_header_row(lines) + 1 :]:
        fields = line.split(";")
        if len(fields) < width:
            skipped += 1
            continue
        if offered_only and "state" in columns and fields[columns["state"]].strip().upper() != "O":
            continue
        try:
            period = int(fields[columns["period"]].strip())
            kind = fields[columns["kind"]].strip().upper()[:1]
            energy = _number(fields[columns["energy"]])
            price = _number(fields[columns["price"]])
        except (ValueError, IndexError):
            skipped += 1
            continue
        if kind in ("C", "V") and np.isfinite(energy) and np.isfinite(price):
            rows.append((period, kind, energy, price))
        else:
            skipped += 1

    if skipped:
        log.debug("%d lines skipped", skipped)
    if not rows:
        raise ValueError("curva_pbc file without usable bids")
    return pd.DataFrame(rows, columns=["period", "kind", "energy_mwh", "price_eur_mwh"])


def build(bids: pd.DataFrame, period: int) -> tuple[Curve, Curve]:
    """Cumulative (buy, sell) curves of one clearing period.

    Bids arrive split into price steps. Supply accumulates from the cheapest
    offer upwards and demand from the dearest bid downwards; both curves are
    stored with increasing prices so they share one representation.
    """
    rows = bids[bids["period"] == period]
    if rows.empty:
        raise ValueError(f"no bids for period {period}")

    built: dict[str, Curve] = {}
    for kind, side in (("V", "sell"), ("C", "buy")):
        lines = rows[rows["kind"] == kind]
        if lines.empty:
            raise ValueError(f"period {period} has no {kind} bids")
        steps = (
            lines.groupby("price_eur_mwh", as_index=False)["energy_mwh"]
            .sum()
            .sort_values("price_eur_mwh", ascending=(side == "sell"))
        )
        price = steps["price_eur_mwh"].to_numpy(float)
        volume = np.cumsum(steps["energy_mwh"].to_numpy(float))
        if side == "buy":
            price, volume = price[::-1], volume[::-1]
        built[side] = Curve(price, volume, side)
    return built["buy"], built["sell"]


def download(days: list[date], dest: Path, pause_s: float = 0.2) -> list[date]:
    """Download the curve files missing from `dest`. Returns the days that failed.

    Mirrors `pce68.data.omie.download`, minus the yearly-zip fallback: OMIE
    serves the curves one day at a time only.
    """
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    with requests.Session() as s:
        for day in days:
            path = dest / file_name(day)
            if path.exists() and path.stat().st_size > 0:
                continue
            try:
                r = s.get(URL.format(day), timeout=60)
                r.raise_for_status()
                text = r.content.decode(ENCODING, errors="replace")
                parse(text)  # never save an error page under a valid file name
                path.write_text(text, encoding=ENCODING)
                time.sleep(pause_s)
            except Exception as e:  # noqa: BLE001 - report it and carry on with the rest
                log.warning("%s: %s", day, e)
                failed.append(day)
    return failed


def load(source: Path, days: list[date] | None = None) -> dict[tuple[date, int], tuple[Curve, Curve]]:
    """Read curve files into the mapping `CurveImpact` expects."""
    curves: dict[tuple[date, int], tuple[Curve, Curve]] = {}
    for path in sorted(source.glob("curva_pbc_*.1")):
        stamp = path.stem.split("_")[-1]
        day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        if days is not None and day not in days:
            continue
        bids = parse(path.read_text(encoding=ENCODING, errors="replace"))
        for period in sorted(bids["period"].unique()):
            try:
                curves[(day, int(period))] = build(bids, int(period))
            except ValueError as e:  # noqa: PERF203 - a bad period must not lose the day
                log.warning("%s period %s: %s", day, period, e)
    return curves
