"""Aggregated supply and demand curves of the day-ahead auction.

The price-taker battery of `pce68.dispatch.battery` assumes the battery does
not move the price. That holds at today's 0.1 GW and fails at the 17 GW of the
2030 NECP. To know by how much it fails we need the price as a function of the
volume a fleet adds, and that function is not a free parameter: it is readable
from the curves OMIE publishes for every auction.

The method is the non-parametric one of Gottfried and Muller (2026): a battery
charging is extra demand, a battery discharging is extra supply, so the
corresponding curve shifts in parallel by the battery's volume and the market
clears again at the new intersection. Nothing is assumed about the shape of the
relation between price and volume; it is read off the curve, which is why the
result is strongly non-linear and differs for every period.

    Gottfried, T. and Muller, G. (2026). Future scenarios for the benefit of
    battery storage in the German day-ahead electricity market. Energy Policy
    217, 115461. doi:10.1016/j.enpol.2026.115461

Sign convention: charging adds demand and pushes the price up; discharging adds
supply and pushes it down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Day-ahead technical limits of the single intraday coupling (SDAC).
PRICE_FLOOR = -500.0
PRICE_CAP = 4000.0


@dataclass(frozen=True)
class Curve:
    """Cumulative aggregated curve of one clearing period, as a step function.

    `price` holds the breakpoints in ascending order and `volume` the cumulative
    MWh at each of them. For `side="sell"` the cumulative volume is the energy
    offered at or below that price, and it grows with price; for `side="buy"` it
    is the energy bid at or above it, and it falls with price.
    """

    price: np.ndarray
    volume: np.ndarray
    side: str

    def __post_init__(self) -> None:
        if self.price.ndim != 1 or self.volume.ndim != 1:
            raise ValueError("price and volume must be one-dimensional")
        if self.price.size != self.volume.size:
            raise ValueError("price and volume must have the same length")
        if self.price.size == 0:
            raise ValueError("an empty curve cannot be cleared")
        if np.any(np.diff(self.price) <= 0):
            raise ValueError("price breakpoints must be strictly increasing")
        if self.side not in ("sell", "buy"):
            raise ValueError("side must be 'sell' or 'buy'")

    def at(self, price: float | np.ndarray) -> np.ndarray:
        """Cumulative volume offered (or bid) at `price`."""
        price = np.asarray(price, dtype=float)
        if self.side == "sell":
            i = np.searchsorted(self.price, price, side="right") - 1
            return np.where(i < 0, 0.0, self.volume[np.clip(i, 0, None)])
        i = np.searchsorted(self.price, price, side="left")
        inside = i < self.price.size
        return np.where(inside, self.volume[np.clip(i, None, self.price.size - 1)], 0.0)

    def shift(self, mwh: float) -> Curve:
        """Parallel shift by `mwh`, the bid of a price taker at every price."""
        if mwh < 0:
            raise ValueError("mwh must be non-negative; shift the other side instead")
        return Curve(self.price.copy(), self.volume + mwh, self.side)


def clear(buy: Curve, sell: Curve) -> tuple[float, float]:
    """Clearing price (EUR/MWh) and volume (MWh) of a pair of curves.

    Excess supply `sell(p) - buy(p)` never decreases with price, so the market
    clears at the lowest breakpoint where it stops being negative.
    """
    grid = np.unique(np.concatenate([buy.price, sell.price]))
    excess = sell.at(grid) - buy.at(grid)

    covered = np.flatnonzero(excess >= 0)
    if covered.size == 0:
        # Demand outruns supply over the whole range: the auction hits the cap.
        return PRICE_CAP, float(sell.at(PRICE_CAP))

    price = float(np.clip(grid[covered[0]], PRICE_FLOOR, PRICE_CAP))
    return price, float(min(sell.at(price), buy.at(price)))


def price_impact(buy: Curve, sell: Curve, mwh: float, action: str) -> float:
    """Change in the clearing price when a battery adds `mwh` to the auction.

    `action` is "charge" (extra demand, price rises) or "discharge" (extra
    supply, price falls). Returns the new price minus the old one.
    """
    base, _ = clear(buy, sell)
    if action == "charge":
        moved, _ = clear(buy.shift(mwh), sell)
    elif action == "discharge":
        moved, _ = clear(buy, sell.shift(mwh))
    else:
        raise ValueError("action must be 'charge' or 'discharge'")
    return moved - base


def impact_curve(buy: Curve, sell: Curve, volumes_mwh: np.ndarray, action: str) -> np.ndarray:
    """Price impact over a grid of volumes.

    Evaluating it volume by volume is the whole point: the merit order steepens
    as it is climbed, so no single slope describes it.
    """
    return np.array([price_impact(buy, sell, float(q), action) for q in np.asarray(volumes_mwh, float)])


def equivalent_slope(buy: Curve, sell: Curve, mwh: float, action: str = "discharge") -> float:
    """Local slope in EUR/MWh per GWh added, as a diagnostic only.

    Reported for comparison with published figures (Germany averaged
    -15.34 EUR/MWh per GWh of added demand in 2024). It is derived from the
    curves and valid only near `mwh`; it is never an input to the simulation.
    """
    if mwh <= 0:
        raise ValueError("mwh must be positive")
    return price_impact(buy, sell, mwh, action) / (mwh / 1000.0)
