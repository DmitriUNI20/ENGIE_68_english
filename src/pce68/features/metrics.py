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


def extreme_cost(p: np.ndarray, h: np.ndarray, energy: float, expensive: bool) -> float:
    """€ for `energy` MWh at 1 MW in the dearest (or cheapest) periods; NaN if they do not hold it.

    Periods are taken from the dearest (or cheapest) down until `energy` is
    reached; the last one only partly. Used both for the daily spread and for
    the battery's charge and discharge.
    """
    if h.sum() < energy - 1e-9:
        return np.nan
    order = np.argsort(-p if expensive else p, kind="stable")
    p, h = p[order], h[order]
    taken = np.clip(energy - (np.cumsum(h) - h), 0, h)  # how much of each period falls inside the window
    return float((p * taken).sum())


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Mean weighted by `weights` (period durations, so an hour counts as four quarter hours)."""
    return float(np.average(values, weights=weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Standard deviation weighted by `weights`."""
    m = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - m) ** 2, weights=weights)))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Value below which half of the total weight lies."""
    order = np.argsort(values, kind="stable")
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


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
            s = (
                extreme_cost(p, h, hours, expensive=True) - extreme_cost(p, h, hours, expensive=False)
            ) / hours
        rows.append((d, s))
    return pd.DataFrame(rows, columns=["day_local", col])
