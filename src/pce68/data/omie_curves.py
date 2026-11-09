"""OMIE aggregated day-ahead curves (curva_pbc files).

Every day OMIE publishes curva_pbc_YYYYMMDD.1: for each clearing period, every
bid and offer as a price step. The layout has changed over time, so columns
are found by name, not by position. Known layouts:

    to 18/03/2025     Hora;...;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;
                      Ofertada (O)/Casada (C);
    from 19/03/2025   Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;
                      Precio Compra/Venta;Ofertada (O)/Casada (C);Tipología de Oferta;
    from 01/10/2025   the same, with quarter-hour periods H1Q1 ... H24Q4

Rows: side (C = purchase/demand, V = sale/supply), quantity, €/MWh, status
(O = offered, C = matched) and, from 2025, bid type (S = simple, C01/C02/C04
= complex conditions, Imp FR / Exp FR = flows with France). Spanish number
format ("1.500,00"). Quantities are read as MW: on hourly days the old MWh
figures are the same number, and the fleet adds MW as well.

The O rows are the curves as bid; the C rows are only what was matched (in
each period matched supply equals matched demand, so they are not a curve).
Crossing the raw O curves misses what EUPHEMIA decides on its own, which
complex bids are accepted and how much flows to and from France: on two real
days of November 2026 that is off by 10-19 €/MWh. The effective curve keeps
the offered simple bids and adds, from the matched rows, the accepted complex
bids and the French flows, which brings it within 2 €/MWh; the rest is closed
per period by `market.curves.calibrate`. Files without a bid-type column
(before 19/03/2025) can only use the offered rows, and lean on calibration.

A raw file weighs ~19 MB. `compact` keeps only what the simulation needs, the
effective curve aggregated by price plus the matched price of each period,
which is a few hundred KB.
"""

from __future__ import annotations

import io
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pce68.market.curves import Curves

URL = "https://www.omie.es/es/file-download?parents=curva_pbc&filename=curva_pbc_{:%Y%m%d}.1"
ENCODING = "latin-1"

# Fragments that identify each column whatever the layout. "tipo oferta" is
# matched in full so it cannot take "tipologia de oferta".
WANTED = {
    "period": ("periodo", "hora"),
    "side": ("tipo oferta",),
    "quantity": ("energ", "potencia"),
    "price": ("precio",),
    "status": ("ofertada",),
    "kind": ("tipologia",),
}
_PERIOD = re.compile(r"^(?:H)?(\d+)(?:Q(\d))?$")


def file_name(day: date) -> str:
    return f"curva_pbc_{day:%Y%m%d}.1"


def compact_path(dest: Path, day: date) -> Path:
    return dest / f"curves_{day:%Y%m%d}.parquet"


def _fold(text: str) -> str:
    """Lowercase without accents or padding, so field names compare reliably."""
    s = unicodedata.normalize("NFKD", text)
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _is_header(line: str) -> bool:
    folded = _fold(line)
    return "precio" in folded and ("energ" in folded or "potencia" in folded)


def period_index(label: str) -> tuple[int, bool]:
    """'H1Q1' -> (1, True), 'H24Q4' -> (96, True), 'H5' or '5' -> (5, False)."""
    m = _PERIOD.match(label.strip())
    if not m:
        raise ValueError(f"unknown period label: {label!r}")
    hour, quarter = int(m.group(1)), m.group(2)
    return (hour, False) if quarter is None else ((hour - 1) * 4 + int(quarter), True)


def _number(s: pd.Series) -> pd.Series:
    cleaned = s.str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_curves(text: str) -> pd.DataFrame:
    """One row per bid step: period, quarter_hourly, side, mw, price, status, kind."""
    lines = text.splitlines()
    header_at = next((i for i, ln in enumerate(lines[:20]) if _is_header(ln)), None)
    if header_at is None:
        raise ValueError("no header found; check the OMIE file layout")
    cols: dict[str, int] = {}
    for i, field in enumerate(_fold(f) for f in lines[header_at].split(";")):
        for name, fragments in WANTED.items():
            if field and name not in cols and any(f in field for f in fragments):
                cols[name] = i
                break
    missing = {"period", "side", "quantity", "price"} - cols.keys()
    if missing:
        raise ValueError(f"OMIE header is missing {sorted(missing)}")

    body = "\n".join(ln for ln in lines[header_at + 1 :] if ln.strip().strip(";"))
    raw = (
        pd.read_csv(io.StringIO(body), sep=";", header=None, dtype=str, keep_default_na=False)
        if body
        else None
    )
    if raw is not None:
        raw = raw[raw[cols["period"]].str.strip().str.fullmatch(_PERIOD.pattern[1:-1])]
    if raw is None or raw.empty:
        raise ValueError("curva_pbc file without bids")

    periods = raw[cols["period"]].map(period_index)
    bids = pd.DataFrame(
        {
            "period": periods.str[0].astype(int),
            "quarter_hourly": periods.str[1].astype(bool),
            "side": raw[cols["side"]].str.strip().str.upper().map({"V": "supply", "C": "demand"}),
            "mw": _number(raw[cols["quantity"]]),
            "price": _number(raw[cols["price"]]),
            "status": (
                raw[cols["status"]].str.strip().str.upper().map({"O": "offered", "C": "matched"})
                if "status" in cols
                else "offered"
            ),
            "kind": raw[cols["kind"]].str.strip() if "kind" in cols else "",
        }
    )
    return bids.dropna(subset=["side", "mw", "price", "status"]).reset_index(drop=True)


def compact(bids: pd.DataFrame) -> pd.DataFrame:
    """What the simulation needs from a parsed file, a small fraction of its size.

    Rows of the effective curve aggregated by (period, side, price), plus one
    row per period with side="matched" holding the matched price (the highest
    matched supply price: the Iberian clearing price). `hours` is each
    period's length.
    """
    offered = bids["status"] == "offered"
    if (bids["kind"] != "").any():
        effective = (offered & (bids["kind"] == "S")) | (
            (bids["status"] == "matched") & (bids["kind"] != "S")
        )
    else:
        effective = offered
    curve = bids[effective].groupby(["period", "side", "price"], as_index=False)["mw"].sum()
    matched = bids[(bids["status"] == "matched") & (bids["side"] == "supply")]
    clearing = matched.groupby("period", as_index=False)["price"].max().assign(side="matched", mw=0.0)
    out = pd.concat([curve, clearing], ignore_index=True)
    hours = bids.groupby("period")["quarter_hourly"].first().map({True: 0.25, False: 1.0})
    out["hours"] = out["period"].map(hours)
    return out.sort_values(["period", "side", "price"], ignore_index=True)


def effective_curves(day: pd.DataFrame) -> list[Curves]:
    """Curves per period, in time order, from a `compact` frame."""
    curves = []
    for _, g in day[day["side"] != "matched"].groupby("period", sort=True):
        s, d = g[g["side"] == "supply"], g[g["side"] == "demand"]
        curves.append(Curves(s["price"], s["mw"], d["price"], d["mw"]))
    return curves


def matched_price(day: pd.DataFrame) -> np.ndarray:
    """Matched (Iberian) price per period, from a `compact` frame. NaN where none was matched."""
    periods = np.sort(day["period"].unique())
    m = day[day["side"] == "matched"].set_index("period")["price"]
    return m.reindex(periods).to_numpy(dtype=float)


def period_hours(day: pd.DataFrame) -> np.ndarray:
    """Length of each period in hours, from a `compact` frame."""
    return day.groupby("period", sort=True)["hours"].first().to_numpy()


def download_raw(day: date, session=None) -> str:
    """Fetch one day's curva_pbc file and return its text (checked to hold bids)."""
    import requests

    get = session.get if session is not None else requests.get
    r = get(URL.format(day), timeout=120)
    r.raise_for_status()
    text = r.content.decode(ENCODING, errors="replace")
    parse_curves(text)  # an error page must not pass for a curve file
    return text
