"""A fleet of batteries that moves the price it trades at.

`pce68.dispatch.battery` answers what one battery earns while it is small
enough to ignore. This module answers what happens once it is not: the fleet
charges in the cheap periods and discharges in the dear ones, which lifts the
first and depresses the second, and the spread it lives on closes behind it.
That is self-cannibalisation, and without this feedback no simulation can
produce it -- a price taker scaled up just multiplies its revenue by the
megawatts, however many are installed.

Entry is sequential, following Gottfried and Muller (2026): the fleet is built
in tranches, and each tranche optimises against the prices left by the ones
before it. This needs no fixed-point iteration and cannot oscillate, because
every tranche faces a market that already contains its predecessors.

What a tranche earns on entry is not what it ends up earning. A uniform-price
auction pays everyone the clearing price, so the whole fleet is settled at the
final one; that is the figure reported, and the gap between the two is the
cannibalisation the last entrant inflicts on the ones already there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from pce68.dispatch.battery import Battery, optimal_cycle_schedule
from pce68.market.curves import Curve, price_impact

TZ = "Europe/Madrid"


class PriceImpact(Protocol):
    """How much the price of one period moves when the fleet adds `mwh` to it."""

    def __call__(self, day: date, period: int, mwh: float, action: str) -> float: ...


@dataclass(frozen=True)
class LinearImpact:
    """A single slope for every period. Provisional, and deliberately crude.

    It exists so the feedback machinery can be exercised before the OMIE curves
    are downloaded, and so the cost of the simplification can be measured
    against `CurveImpact` on the same days. The real merit order steepens as it
    is climbed, so one slope under-reacts in scarcity and over-reacts in the
    solar trough. Any absolute figure obtained with this is provisional.
    """

    slope_eur_mwh_per_gwh: float

    def __call__(self, day: date, period: int, mwh: float, action: str) -> float:
        delta = self.slope_eur_mwh_per_gwh * mwh / 1000.0
        return delta if action == "charge" else -delta


@dataclass(frozen=True)
class CurveImpact:
    """Price impact read off the aggregated curves OMIE publishes per period.

    `curves` maps (local day, period) to that period's (buy, sell) curve.
    Periods without a curve get no impact, which keeps a partial download
    usable while making the omission visible through `missing`.
    """

    curves: dict[tuple[date, int], tuple[Curve, Curve]]

    def __call__(self, day: date, period: int, mwh: float, action: str) -> float:
        pair = self.curves.get((day, period))
        if pair is None:
            return 0.0
        return price_impact(pair[0], pair[1], mwh, action)

    def missing(self, days: list[date], periods: int) -> int:
        """How many (day, period) pairs of a span have no curve loaded."""
        return sum(1 for d in days for h in range(1, periods + 1) if (d, h) not in self.curves)


@dataclass
class FleetResult:
    """What the fleet did and what it left behind.

    `daily` has one row per local day with the fleet's revenue per installed MW
    and the energy it moved. `prices` is the price series after the feedback,
    ready for `daily_spread` or any other metric, with the same columns as the
    input so it is a drop-in replacement.
    """

    daily: pd.DataFrame
    prices: pd.DataFrame
    fleet_mw: float


def simulate_fleet(
    prices: pd.DataFrame,
    bat: Battery,
    impact: PriceImpact,
    fleet_mw: float,
    tranche_mw: float = 250.0,
) -> FleetResult:
    """Dispatch `fleet_mw` of batteries against a price series that answers back.

    `prices` needs timestamp_utc, price_es and resolution_min, as everywhere
    else in the project. `tranche_mw` is the size of each sequential entrant;
    smaller is more faithful and slower, and 250 MW matches the 0.25 GWh agents
    of the reference at a 2 h duration.

    The annual cycle cap is applied as in `simulate_price_taker`. Note that the
    feedback is computed on every profitable day, so if the cap ever binds the
    returned prices reflect slightly more activity than the capped dispatch; at
    the configured 365 cycles the cap is inert.
    """
    if fleet_mw <= 0:
        raise ValueError("fleet_mw must be positive")
    if tranche_mw <= 0:
        raise ValueError("tranche_mw must be positive")

    local_day = prices["timestamp_utc"].dt.tz_convert(TZ).dt.date
    moved = prices.copy()
    rows = []

    for day, group in prices.groupby(local_day, sort=True):
        group = group.sort_values("timestamp_utc")
        index = group.index
        p = group["price_es"].to_numpy(float).copy()
        h = group["resolution_min"].to_numpy(float) / 60.0

        charged = np.zeros_like(p)
        discharged = np.zeros_like(p)
        remaining = fleet_mw

        while remaining > 1e-9:
            step_mw = min(tranche_mw, remaining)
            charge_1mw, discharge_1mw, margin = optimal_cycle_schedule(p, h, bat)
            if margin <= 0:
                break  # the spread is gone: no further tranche would enter

            charge = charge_1mw * step_mw
            discharge = discharge_1mw * step_mw
            charged += charge
            discharged += discharge
            remaining -= step_mw

            net = charge - discharge
            for t in np.flatnonzero(np.abs(net) > 1e-9):
                action = "charge" if net[t] > 0 else "discharge"
                p[t] += impact(day, int(t) + 1, abs(float(net[t])), action)

        moved.loc[index, "price_es"] = p
        revenue = float((p * discharged).sum() - (p * charged).sum())
        rows.append((day, revenue / fleet_mw, float(charged.sum()), float(discharged.sum())))

    daily = pd.DataFrame(rows, columns=["day", "revenue_eur_per_mw", "charged_mwh", "discharged_mwh"])
    daily["cycles"] = (daily["discharged_mwh"] > 0).astype(int)

    year = pd.to_datetime(daily["day"]).dt.year
    rank = daily.groupby(year)["revenue_eur_per_mw"].rank(method="first", ascending=False)
    daily.loc[rank > bat.max_cycles_per_year, ["revenue_eur_per_mw", "cycles"]] = 0

    return FleetResult(daily=daily, prices=moved, fleet_mw=fleet_mw)


def cannibalisation_curve(
    prices: pd.DataFrame,
    bat: Battery,
    impact: PriceImpact,
    fleet_sizes_mw: list[float],
    tranche_mw: float = 250.0,
) -> pd.DataFrame:
    """Revenue per MW and resulting spread for a range of fleet sizes.

    This is the table the project is for. Revenue per MW falling as the fleet
    grows is self-cannibalisation; read against the annualised cost of storage
    it gives the size at which building more stops paying.

    The first row is the price-taker case (`fleet_mw` of zero), which is the
    counterfactual the whole comparison hangs on: with roughly 0.1 GW installed
    in Spain, the observed series is already close to it, which is why the
    Spanish market is a cleaner laboratory for this than the German one.
    """
    from pce68.features.metrics import daily_spread

    base_spread = daily_spread(prices, bat.duration_hours)[f"spread_{bat.duration_hours:g}h"]
    rows = [
        {
            "fleet_mw": 0.0,
            "revenue_eur_per_mw_year": float(
                _annualised(simulate_fleet(prices, bat, LinearImpact(0.0), 1.0).daily)
            ),
            "mean_spread_eur_mwh": float(base_spread.mean()),
            "mean_price_eur_mwh": _weighted_mean(prices),
        }
    ]
    for size in fleet_sizes_mw:
        result = simulate_fleet(prices, bat, impact, size, tranche_mw)
        spread = daily_spread(result.prices, bat.duration_hours)[f"spread_{bat.duration_hours:g}h"]
        rows.append(
            {
                "fleet_mw": size,
                "revenue_eur_per_mw_year": float(_annualised(result.daily)),
                "mean_spread_eur_mwh": float(spread.mean()),
                "mean_price_eur_mwh": _weighted_mean(result.prices),
            }
        )
    out = pd.DataFrame(rows)
    out["revenue_vs_price_taker"] = out["revenue_eur_per_mw_year"] / out["revenue_eur_per_mw_year"].iloc[0]
    return out


def _annualised(daily: pd.DataFrame) -> float:
    """Revenue per MW scaled to a full year, so spans of different length compare."""
    if daily.empty:
        return 0.0
    days = max(len(daily), 1)
    return daily["revenue_eur_per_mw"].sum() * 365.0 / days


def _weighted_mean(prices: pd.DataFrame) -> float:
    """Mean price weighted by each period's duration, so hours and quarter hours count alike."""
    weight = prices["resolution_min"] / 60.0
    return float((prices["price_es"] * weight).sum() / weight.sum())
