"""Day-ahead dispatch of a battery fleet large enough to move prices.

Unlike `battery.py` (one price-taking MW), a fleet of X GW changes the prices
it trades at: charging lifts the trough, discharging lowers the peak. Charging
the whole fleet in the cheapest hours of the original prices would turn those
hours into new peaks, so the schedule is built incrementally:

1. Split the fleet's daily energy E = X · duration into blocks of fixed size
   (100 MWh by default), the same for every fleet so that results for
   different X are comparable.
2. For each block, pick the charge period i and later discharge period j
   with the best margin at the *current* prices: sell b MWh at p_j, buy
   b / efficiency MWh at p_i.
3. Add the block, re-clear both periods, repeat.
4. Stop when no pair still makes money or the fleet has done one full cycle.

This is how a competitive fleet behaves in aggregate: arbitrage continues
until the spread no longer covers the losses (p_j · efficiency <= p_i).

Constraints: power <= X in every period, no charging and discharging in the
same period, each discharged block charged earlier the same day (so the state
of charge never goes negative and never exceeds E), at most one full cycle a
day. Perfect foresight of the day's curves, as in `battery.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pce68.market.curves import Curves, clear


@dataclass(frozen=True)
class Fleet:
    power_mw: float
    duration_hours: float
    round_trip_efficiency: float

    def __post_init__(self) -> None:
        if self.power_mw < 0:
            raise ValueError("power_mw must be non-negative")
        if self.duration_hours <= 0:
            raise ValueError("duration_hours must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")


def dispatch_fleet(
    curves: list[Curves], hours: np.ndarray, fleet: Fleet, block_mwh: float = 100.0
) -> pd.DataFrame:
    """Schedule the fleet over one day and return the prices it leaves behind.

    `curves` and `hours` hold one entry per period, in time order (`hours` is
    each period's length: 1 or 0.25). Returns one row per period: base_price
    (no fleet), price (with fleet), charge_mw, discharge_mw.
    """
    hours = np.asarray(hours, dtype=float)
    n = len(curves)
    if len(hours) != n:
        raise ValueError("one duration per period is needed")
    base = np.array([clear(c) for c in curves])
    price = base.copy()
    charge = np.zeros(n)
    discharge = np.zeros(n)

    energy = fleet.power_mw * fleet.duration_hours
    if energy > 0 and n > 1:
        eta = fleet.round_trip_efficiency
        # A block must fit in one period at full power, charging losses included.
        block = min(block_mwh, energy, fleet.power_mw * hours.min() * eta)
        cap = fleet.power_mw + 1e-9
        pairs: list[tuple[int, int]] = []  # (charge period, discharge period) of every block
        sold = 0.0
        while sold < energy - 1e-9:
            can_charge = (discharge == 0) & (charge + block / eta / hours <= cap)
            can_discharge = (charge == 0) & (discharge + block / hours <= cap)
            # For each j, the cheapest eligible charge period strictly before it.
            before = np.full(n, -1)
            best = -1
            for t in range(n):
                before[t] = best
                if can_charge[t] and (best < 0 or price[t] < price[best]):
                    best = t
            cost = np.where(before >= 0, price[before], np.inf)
            margin = np.where(can_discharge, block * price - block / eta * cost, -np.inf)

            # Try candidates from the best margin down. A block is accepted only if,
            # at the prices it leaves behind, every block charged or discharged in
            # those two periods still pays: price_j · efficiency >= price_i. The
            # market price applies to all the energy in a period, not just the last
            # block, so checking before the move would let the fleet trade at a loss.
            accepted = False
            for j in np.argsort(-margin):
                if margin[j] <= 1e-9:
                    break
                i = int(before[j])
                new_charge = charge[i] + block / eta / hours[i]
                new_discharge = discharge[j] + block / hours[j]
                trial = price.copy()
                trial[i] = clear(curves[i], extra_demand_mw=new_charge)
                trial[j] = clear(curves[j], extra_supply_mw=new_discharge)
                touched = [(a, b) for a, b in [*pairs, (i, j)] if {a, b} & {i, j}]
                if all(trial[b] * eta >= trial[a] - 1e-9 for a, b in touched):
                    charge[i], discharge[j], price = new_charge, new_discharge, trial
                    pairs.append((i, int(j)))
                    sold += block
                    accepted = True
                    break
            if not accepted:
                break

    return pd.DataFrame({"base_price": base, "price": price, "charge_mw": charge, "discharge_mw": discharge})


def fleet_revenue(day: pd.DataFrame, hours: np.ndarray) -> float:
    """€ earned by the whole fleet in the day, at the prices it leaves behind."""
    net_mwh = (day["discharge_mw"] - day["charge_mw"]).to_numpy() * np.asarray(hours, dtype=float)
    return float((net_mwh * day["price"].to_numpy()).sum())
