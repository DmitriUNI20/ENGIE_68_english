"""Self-cannibalisation curve: what a fleet earns as it grows.

    uv run python scripts/05_fleet.py        # or `make fleet`

Summarises the panel written by 03_simulate_impact.py into one row per fleet
size: revenue per installed MW (annualised), how much of the small-fleet
revenue it keeps, the mean daily spread it leaves behind and the mean price.
Revenue per MW falling as the fleet grows is the effect the project is about;
read against the annualised cost of storage it gives the size at which
building more stops paying.

All prices are the ones the fleet leaves behind (uniform-price auction: the
whole fleet is paid the final clearing price), weighted by period length.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def cannibalisation_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per fleet size from the impact panel (one row per day, period and size)."""
    p = panel.assign(
        revenue=(panel["discharge_mw"] - panel["charge_mw"]) * panel["hours"] * panel["p_x"],
        weighted_price=panel["p_x"] * panel["hours"],
    )
    days = p["day"].nunique()
    by_day = p.groupby(["bess_gw", "day"]).agg(
        revenue=("revenue", "sum"), low=("p_x", "min"), high=("p_x", "max")
    )
    by_day["spread"] = by_day["high"] - by_day["low"]
    table = by_day.groupby("bess_gw").agg(revenue=("revenue", "sum"), spread=("spread", "mean"))
    table["mean_price"] = p.groupby("bess_gw")["weighted_price"].sum() / p.groupby("bess_gw")["hours"].sum()
    gw = table.index.to_series()
    table["revenue_eur_per_mw_year"] = (table["revenue"] / (gw * 1000)).where(gw > 0) * 365 / days
    first = table["revenue_eur_per_mw_year"].dropna().iloc[0]
    table["vs_smallest_fleet"] = table["revenue_eur_per_mw_year"] / first
    return table.reset_index()[
        ["bess_gw", "revenue_eur_per_mw_year", "vs_smallest_fleet", "spread", "mean_price"]
    ]


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    impact_dir = ROOT / config["impact"]["output_dir"]
    files = sorted(impact_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no simulated days in {impact_dir}; run scripts/03_simulate_impact.py first.")
    panel = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    table = cannibalisation_table(panel)

    days = panel["day"].nunique()
    log.info("%d days, %s to %s\n", days, panel["day"].min().date(), panel["day"].max().date())
    print(f"{'fleet':>6}  {'revenue':>14}  {'vs smallest':>11}  {'daily spread':>12}  {'mean price':>10}")
    print(f"{'GW':>6}  {'EUR/MW/year':>14}  {'fleet':>11}  {'EUR/MWh':>12}  {'EUR/MWh':>10}")
    for r in table.itertuples():
        revenue = (
            f"{r.revenue_eur_per_mw_year:>14,.0f}" if pd.notna(r.revenue_eur_per_mw_year) else f"{'-':>14}"
        )
        keeps = f"{100 * r.vs_smallest_fleet:>10.0f}%" if pd.notna(r.vs_smallest_fleet) else f"{'-':>11}"
        print(f"{r.bess_gw:>6g}  {revenue}  {keeps}  {r.spread:>12.2f}  {r.mean_price:>10.2f}")

    destination = ROOT / config["data"]["processed_dir"] / "fleet.csv"
    table.to_csv(destination, index=False)
    log.info("\nWrote %s", destination)


if __name__ == "__main__":
    main()
