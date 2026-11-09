"""Aggregated supply and demand curves of one market period, and their clearing.

A period's curves are the price-quantity steps of every bid:

    supply  S(p) = MW offered at prices <= p      (non-decreasing in p)
    demand  D(p) = MW bid at prices >= p          (non-increasing in p)

The clearing price is where they cross. With step curves the crossing is a
step, so the convention here is: the lowest bid price p at which supply covers
the demand bidding strictly above it, S(p) >= D(p+). Whichever side's step
sits at that price sets it: a supply step (the demand is still willing to pay
more) or a demand step (the supply has spare MW).

This is a simplification of EUPHEMIA, which also handles block orders,
complex conditions and market coupling, so the reconstructed price must be
checked against the published one before it is trusted (`reconstruction_error`).

A battery fleet enters as price-taking quantities: charging is demand bid at
the price cap (it buys whatever the price), discharging is supply offered at
the price floor (it sells whatever the price). Both shift a curve sideways.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRICE_FLOOR = -500.0  # €/MWh, harmonised day-ahead minimum
PRICE_CAP = 4000.0  # €/MWh, harmonised day-ahead maximum


@dataclass(frozen=True)
class Curves:
    """Bid steps of one period. Quantities in MW, prices in €/MWh."""

    supply_price: np.ndarray
    supply_mw: np.ndarray
    demand_price: np.ndarray
    demand_mw: np.ndarray

    def __post_init__(self) -> None:
        for name in ("supply_price", "supply_mw", "demand_price", "demand_mw"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        if len(self.supply_price) != len(self.supply_mw) or len(self.demand_price) != len(self.demand_mw):
            raise ValueError("each price array needs a quantity array of the same length")
        if (self.supply_mw < 0).any() or (self.demand_mw < 0).any():
            raise ValueError("bid quantities must be non-negative")


def _cumulative(prices: np.ndarray, mw: np.ndarray, at: np.ndarray, side: str) -> np.ndarray:
    """MW bid at prices <= `at` (side='supply') or strictly above `at` (side='demand')."""
    order = np.argsort(prices)
    p, q = prices[order], mw[order]
    cum = np.concatenate([[0.0], np.cumsum(q)])
    if side == "supply":
        return cum[np.searchsorted(p, at, side="right")]
    return cum[-1] - cum[np.searchsorted(p, at, side="right")]


def clear(
    curves: Curves,
    extra_demand_mw: float = 0.0,
    extra_supply_mw: float = 0.0,
    floor: float = PRICE_FLOOR,
    cap: float = PRICE_CAP,
) -> float:
    """Clearing price of one period, optionally with price-taking extra demand and supply.

    Returns `floor` if supply covers demand at any price, and `cap` if it
    never does (scarcity).
    """
    candidates = np.unique(np.concatenate([curves.supply_price, curves.demand_price, [floor, cap]]))
    candidates = candidates[(candidates >= floor) & (candidates <= cap)]
    supply = _cumulative(curves.supply_price, curves.supply_mw, candidates, "supply") + extra_supply_mw
    demand = _cumulative(curves.demand_price, curves.demand_mw, candidates, "demand") + extra_demand_mw
    covered = np.flatnonzero(supply >= demand - 1e-9)
    return float(candidates[covered[0]]) if covered.size else float(cap)


def calibrate(
    curves: Curves, target_price: float, floor: float = PRICE_FLOOR, cap: float = PRICE_CAP
) -> float:
    """Demand shift (MW) that makes `curves` clear at `target_price`.

    Reconstructed curves never match EUPHEMIA exactly, so each period is
    nudged sideways until it reproduces the published price; with no battery
    the simulator then returns the historical price, as it must. Any shift
    in an interval does the job; the middle of it is returned so that small
    battery volumes move the price the way the real curve would. The target
    snaps to the nearest bid price, the only prices step curves can clear at.
    """
    candidates = np.unique(np.concatenate([curves.supply_price, curves.demand_price, [floor, cap]]))
    candidates = candidates[(candidates >= floor) & (candidates <= cap)]
    surplus = _cumulative(curves.supply_price, curves.supply_mw, candidates, "supply") - _cumulative(
        curves.demand_price, curves.demand_mw, candidates, "demand"
    )
    k = int(np.abs(candidates - target_price).argmin())
    high = surplus[k]  # shift <= high: supply still covers demand at the target
    low = surplus[k - 1] if k > 0 else high - 1.0  # shift > low: not yet covered one step below
    return float((low + high) / 2) if high > low else float(high)


def shifted(curves: Curves, demand_mw: float) -> Curves:
    """`curves` with `demand_mw` of price-taking demand added (negative: supply added)."""
    if demand_mw >= 0:
        return Curves(
            curves.supply_price,
            curves.supply_mw,
            np.append(curves.demand_price, PRICE_CAP),
            np.append(curves.demand_mw, demand_mw),
        )
    return Curves(
        np.append(curves.supply_price, PRICE_FLOOR),
        np.append(curves.supply_mw, -demand_mw),
        curves.demand_price,
        curves.demand_mw,
    )


def calibrated(curves: list[Curves], prices: np.ndarray) -> list[Curves]:
    """Each period's curves shifted so that, with nothing added, they clear at `prices`."""
    return [shifted(c, calibrate(c, p)) for c, p in zip(curves, prices, strict=True)]


def reconstruction_error(curves: list[Curves], published: np.ndarray) -> np.ndarray:
    """Reconstructed minus published price per period (€/MWh). Should be ~0 before simulating."""
    return np.array([clear(c) for c in curves]) - np.asarray(published, dtype=float)
