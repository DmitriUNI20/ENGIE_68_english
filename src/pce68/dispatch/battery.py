"""Day-ahead arbitrage of a battery acting as a price taker.

Assumptions (all open to review; they live here, not scattered through the code):

* 1 MW of power; revenue comes out in €/MW and scales linearly.
* Price taker: the battery does not move the price. Reasonable with the
  0.1 GW installed today; not with the 17 GW of the 2030 NECP (PNIEC), and
  this simulation's revenue will then be a ceiling.
* Perfect foresight of the day's prices (known after market clearing at
  13:00 on D-1). Another ceiling: in reality bids go in without knowing them.
* At most one full cycle per day, charging before discharging. With a 2 h
  duration and the Spanish duck curve (solar trough, evening peak) almost all
  the value lies in that cycle.
* Round-trip efficiency applied on charging: to discharge E MWh the battery
  must buy E/efficiency MWh, always at 1 MW at most.
* Annual cycle cap: if a year has more profitable days than allowed cycles,
  the best days are kept.
* Day-ahead energy only. No balancing services, no degradation, no grid
  fees and no operating costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"


@dataclass(frozen=True)
class Battery:
    duration_hours: float
    round_trip_efficiency: float
    max_cycles_per_year: int

    def __post_init__(self) -> None:
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if self.duration_hours <= 0:
            raise ValueError("duration_hours must be positive")


def _extreme_alloc(
    p: np.ndarray, h: np.ndarray, energy: float, expensive: bool
) -> tuple[np.ndarray | None, float]:
    """MWh taken from each period and their € cost, filling the dearest (or cheapest) first.

    The allocation comes back in the original period order, so it can be used
    as a schedule. Returns (None, NaN) when `energy` does not fit.
    """
    if h.sum() < energy - 1e-9:
        return None, np.nan
    order = np.argsort(-p if expensive else p, kind="stable")
    ps, hs = p[order], h[order]
    taken_sorted = np.clip(energy - (np.cumsum(hs) - hs), 0, hs)
    taken = np.zeros_like(p)
    taken[order] = taken_sorted
    return taken, float((ps * taken_sorted).sum())


def _extreme_cost(p: np.ndarray, h: np.ndarray, energy: float, expensive: bool) -> float:
    """€ for moving `energy` MWh at 1 MW in the dearest (or cheapest) periods; NaN if it does not fit."""
    return _extreme_alloc(p, h, energy, expensive)[1]


def optimal_cycle(p: np.ndarray, h: np.ndarray, bat: Battery) -> float:
    """Best margin (€/MW) of one cycle in a day: charge in [0, k), discharge in [k, n).

    For each split k the battery buys in the cheapest periods before k and
    sells in the most expensive ones from k on. It is exact for one cycle with
    charge and discharge at rated power. Returns 0 if no cycle makes money.
    """
    energy_sold = bat.duration_hours
    energy_bought = energy_sold / bat.round_trip_efficiency
    best = 0.0
    for k in range(1, len(p)):
        cost = _extreme_cost(p[:k], h[:k], energy_bought, expensive=False)
        if np.isnan(cost):
            continue
        income = _extreme_cost(p[k:], h[k:], energy_sold, expensive=True)
        if np.isnan(income):
            break  # every later k leaves even fewer periods to discharge in
        best = max(best, income - cost)
    return best


def optimal_cycle_schedule(
    p: np.ndarray, h: np.ndarray, bat: Battery
) -> tuple[np.ndarray, np.ndarray, float]:
    """Same optimum as `optimal_cycle`, but returning what the battery does.

    Needed by `pce68.dispatch.fleet`: to know how a fleet moves the price we
    need the MWh it buys and sells in each period, not just the margin.

    Returns (charge_mwh, discharge_mwh, margin) for 1 MW of power. Both arrays
    are zero when no cycle makes money.
    """
    energy_sold = bat.duration_hours
    energy_bought = energy_sold / bat.round_trip_efficiency
    best = 0.0
    charge = np.zeros_like(p)
    discharge = np.zeros_like(p)
    for k in range(1, len(p)):
        buy, cost = _extreme_alloc(p[:k], h[:k], energy_bought, expensive=False)
        if buy is None:
            continue
        sell, income = _extreme_alloc(p[k:], h[k:], energy_sold, expensive=True)
        if sell is None:
            break  # every later k leaves even fewer periods to discharge in
        if income - cost > best:
            best = income - cost
            charge = np.concatenate([buy, np.zeros(len(p) - k)])
            discharge = np.concatenate([np.zeros(k), sell])
    return charge, discharge, best


def simulate_price_taker(prices: pd.DataFrame, bat: Battery) -> pd.DataFrame:
    """Daily arbitrage revenue of 1 MW of battery.

    `prices` needs timestamp_utc, price_es (€/MWh) and resolution_min.
    Returns one row per local day: day, revenue_eur_per_mw, cycles (0 or 1).
    """
    day = prices["timestamp_utc"].dt.tz_convert(TZ).dt.date
    rows = []
    for d, g in prices.groupby(day, sort=True):
        g = g.sort_values("timestamp_utc")
        margin = optimal_cycle(g["price_es"].to_numpy(float), g["resolution_min"].to_numpy(float) / 60.0, bat)
        rows.append((d, margin))
    out = pd.DataFrame(rows, columns=["day", "revenue_eur_per_mw"])
    out["cycles"] = (out["revenue_eur_per_mw"] > 0).astype(int)

    # Annual cap: outside the best days, the battery does not cycle.
    year = pd.to_datetime(out["day"]).dt.year
    rank = out.groupby(year)["revenue_eur_per_mw"].rank(method="first", ascending=False)
    excluded = rank > bat.max_cycles_per_year
    out.loc[excluded, ["revenue_eur_per_mw", "cycles"]] = 0
    return out
