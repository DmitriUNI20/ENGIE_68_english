"""Daily metrics on the price series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"


def _days(prices: pd.DataFrame):
    """Iterate over (local day, that day's prices, hours of each period)."""
    day = prices["timestamp_utc"].dt.tz_convert(TZ).dt.date
    for d, g in prices.groupby(day, sort=True):
        g = g.sort_values("timestamp_utc")
        yield d, g["price_es"].to_numpy(float), (g["resolution_min"].to_numpy(float) / 60.0)


def _extreme_mean(p: np.ndarray, h: np.ndarray, hours: float, expensive: bool) -> float:
    """Mean price of the day's `hours` most expensive (or cheapest) hours, weighted by time."""
    order = np.argsort(-p if expensive else p, kind="stable")
    p, h = p[order], h[order]
    cum = np.cumsum(h)
    taken = np.clip(hours - (cum - h), 0, h)  # how much of each period falls inside the window
    return float((p * taken).sum() / hours)


def daily_spread(prices: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Capturable spread of each day: mean of the `hours` most expensive minus the cheapest.

    It is the reference metric for arbitrage with a battery of `hours`
    duration: gross, with neither efficiency nor time ordering (charge before
    discharge). The realistic revenue comes from `pce68.dispatch.battery`.

    Returns columns `day_local` and `spread_<hours>h` (e.g. `spread_2h`).
    Days with less than 2·`hours` of data are NaN.
    """
    col = f"spread_{hours:g}h"
    rows = []
    for d, p, h in _days(prices):
        if h.sum() < 2 * hours:
            s = np.nan
        else:
            s = _extreme_mean(p, h, hours, expensive=True) - _extreme_mean(p, h, hours, expensive=False)
        rows.append((d, s))
    return pd.DataFrame(rows, columns=["day_local", col])
