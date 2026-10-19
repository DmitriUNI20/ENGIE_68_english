"""OMIE aggregated supply and demand curves (curva_pbc files).

Alongside the cleared price, OMIE publishes the curves the price came from:
one file per day holding, for every clearing period, the bids and offers
aggregated into price steps. They are public with roughly a ninety-day delay
and the archive is a rolling six years, which is exactly the span this project
covers -- what is not downloaded now is lost.

    curva_pbc_YYYYMMDD.1

The reader finds its columns by name rather than position, because the layout
is not stable. Two changes are known, both from real files:

    to 18/03/2025    Hora;...;Tipo Oferta;Energia Compra/Venta;Precio ...
    from 19/03/2025  Periodo;...;Tipo Oferta;Potencia Compra/Venta;Precio ...;
                     Tipologia de Oferta;

Besides the renaming, the quantity changed meaning: the newer files declare
POWER in MW, not energy in MWh. Over an hourly period the two coincide, but
over a quarter hour 1 MW is 0.25 MWh. Clearing is unaffected, since both curves
move together, but the battery volume added to a curve is in MWh and would be
overstated fourfold if it were not converted. `parse` therefore returns MWh
whatever the file declares.

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
# "tipo oferta" is matched in full so it cannot swallow "tipologia de oferta",
# the column added in 2025, and "quantity" covers both the old energy column
# and the power one that replaced it.
WANTED = {
    "period": ("periodo", "hora"),
    "kind": ("tipo oferta",),
    "quantity": ("energ", "potencia"),
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
    """Index of the header line: the first one naming a price and a quantity."""
    for i, line in enumerate(lines[:20]):
        folded = _fold(line)
        if "precio" in folded and ("energ" in folded or "potencia" in folded):
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
    missing = {"period", "kind", "quantity", "price"} - found.keys()
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
    start = _header_row(lines)
    header = lines[start].split(";")
    columns = _columns(header)
    width = max(columns.values()) + 1
    declares_power = "potencia" in _fold(header[columns["quantity"]])

    rows: list[tuple[int, str, float, float]] = []
    skipped = 0
    for line in lines[start + 1 :]:
        fields = line.split(";")
        if len(fields) < width:
            skipped += 1
            continue
        if offered_only and "state" in columns and fields[columns["state"]].strip().upper() != "O":
            continue
        try:
            period = int(fields[columns["period"]].strip())
            kind = fields[columns["kind"]].strip().upper()[:1]
            quantity = _number(fields[columns["quantity"]])
            price = _number(fields[columns["price"]])
        except (ValueError, IndexError):
            skipped += 1
            continue
        if kind in ("C", "V") and np.isfinite(quantity) and np.isfinite(price):
            rows.append((period, kind, quantity, price))
        else:
            skipped += 1

    if skipped:
        log.debug("%d lines skipped", skipped)
    if not rows:
        raise ValueError("curva_pbc file without usable bids")
    bids = pd.DataFrame(rows, columns=["period", "kind", "energy_mwh", "price_eur_mwh"])

    if declares_power:
        # MW over a period of this length is MWh. Quarter-hourly days hold 92,
        # 96 or 100 periods; hourly ones 23, 24 or 25.
        hours = 0.25 if bids["period"].nunique() > 50 else 1.0
        bids["energy_mwh"] *= hours
    return bids


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
