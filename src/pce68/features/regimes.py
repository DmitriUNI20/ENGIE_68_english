"""Regimes of the Spanish day-ahead market.

Five years of prices are not a single market. Training on the whole series
mixes situations that no longer exist, so every period is labelled with the
regime it belongs to and the models filter on that label.

The boundaries are policy or market dates; they are not derived from the data:

    pre_crisis         up to 30/06/2021. TTF gas within its historical range.
    gas_crisis         01/07/2021 - 14/06/2022. TTF soars in summer 2021
                       (record-low storage, China bidding for LNG, Russia
                       offering no spot gas) and the invasion of Ukraine
                       (24/02/2022) makes it worse. Combined-cycle gas sets the
                       price almost every hour.
    iberian_exception  15/06/2022 - 31/12/2023. Cap on the gas used by plants
                       clearing in the day-ahead market (RDL 10/2022). The OMIE
                       price excludes the compensation paid to gas plants, so
                       the series is artificially low against the real cost.
    normalization      from 01/01/2024. Gas back at 30-40 €/MWh and solar
                       growing faster than demand: zero or negative hours
                       (115 in 2023, 819 in 2024) and the duck curve. This is
                       the market being projected.

One-off events such as the blackout of 28/04/2025 are not a regime: they
change system operation (more combined cycles in technical constraints) but
barely move the day-ahead price.
"""

from __future__ import annotations

import pandas as pd

TZ = "Europe/Madrid"

# (name, first local day included). Each regime lasts until the next one starts.
REGIMES: list[tuple[str, str]] = [
    ("pre_crisis", "1900-01-01"),
    ("gas_crisis", "2021-07-01"),
    ("iberian_exception", "2022-06-15"),
    ("normalization", "2024-01-01"),
]


def annotate(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `prices` with a `regime` column."""
    out = prices.copy()
    local = out["timestamp_utc"].dt.tz_convert(TZ)
    day = local.dt.tz_localize(None).dt.normalize()
    cuts = [pd.Timestamp(d) for _, d in REGIMES[1:]]
    names = [n for n, _ in REGIMES]
    idx = pd.Series(0, index=out.index)
    for cut in cuts:
        idx += (day >= cut).astype(int)
    out["regime"] = pd.Categorical.from_codes(idx.to_numpy(), categories=names, ordered=True)
    return out


def summary(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per regime present in the data, in chronological order.

    Columns: regime, periods, start, end (local days), mean_price, max_price.
    The mean price is weighted by each period's duration, so hours and quarter
    hours count the same per unit of time.
    """
    if "regime" not in prices:
        prices = annotate(prices)
    df = prices.assign(
        _day=prices["timestamp_utc"].dt.tz_convert(TZ).dt.date,
        _weight=prices["resolution_min"] / 60.0,
    )
    df["_pxh"] = df["price_es"] * df["_weight"]
    g = df.groupby("regime", observed=True)
    out = pd.DataFrame(
        {
            "periods": g.size(),
            "start": g["_day"].min(),
            "end": g["_day"].max(),
            "mean_price": g["_pxh"].sum() / g["_weight"].sum(),
            "max_price": g["price_es"].max(),
        }
    )
    return out.reset_index().assign(regime=lambda d: d["regime"].astype(str))
