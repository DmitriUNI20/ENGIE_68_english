"""Historical market-impact simulator: (day, X) -> prices with X GW of storage.

Stage 1 of the BESS cannibalisation roadmap. For one day's OMIE curves:

1. Build the effective curves and calibrate them to the historical price,
   so that with no storage the simulator returns exactly p0.
2. For each fleet size X on the grid, dispatch the fleet against those
   curves and record the prices it leaves behind, p^X.
3. Report the multiplicative impact y = p^X / p0 where p0 is far enough from
   zero for the ratio to mean something, and the additive impact p^X - p0
   everywhere (Spain has hundreds of zero and negative hours a year, where
   the ratio explodes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pce68.data.omie_curves import effective_curves, matched_price, period_hours
from pce68.dispatch.fleet import Fleet, dispatch_fleet
from pce68.market.curves import calibrated


def simulate_day(
    day: pd.DataFrame,
    grid_gw: list[float],
    duration_hours: float,
    round_trip_efficiency: float,
    base_prices: np.ndarray | None = None,
    min_abs_price: float = 5.0,
) -> pd.DataFrame:
    """One row per (period, X): p0, pX, delta, ratio, charge and discharge of the fleet.

    `day` is a `compact` curva_pbc frame. `base_prices` are the historical prices
    to calibrate to (published Spanish prices when available; the matched
    Iberian price from the file otherwise). The ratio is NaN where
    |p0| < `min_abs_price`.
    """
    curves = effective_curves(day)
    hours = period_hours(day)
    p0 = matched_price(day) if base_prices is None else np.asarray(base_prices, dtype=float)
    if len(p0) != len(curves):
        raise ValueError(f"{len(curves)} periods of curves but {len(p0)} prices")
    if np.isnan(p0).any():
        raise ValueError("some periods have no historical price to calibrate to")
    curves = calibrated(curves, p0)

    frames = []
    for gw in grid_gw:
        fleet = dispatch_fleet(curves, hours, Fleet(gw * 1000, duration_hours, round_trip_efficiency))
        frames.append(
            pd.DataFrame(
                {
                    "period": np.arange(1, len(curves) + 1),
                    "hours": hours,
                    "bess_gw": gw,
                    "p0": p0,
                    "p_x": fleet["price"].to_numpy(),
                    "charge_mw": fleet["charge_mw"].to_numpy(),
                    "discharge_mw": fleet["discharge_mw"].to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["delta"] = out["p_x"] - out["p0"]
    out["ratio"] = np.where(out["p0"].abs() >= min_abs_price, out["p_x"] / out["p0"], np.nan)
    return out
