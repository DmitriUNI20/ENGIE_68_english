"""Self-cannibalisation curve: what a fleet earns as it grows.

    uv run python scripts/05_fleet.py --demo          # synthetic days, no data needed
    uv run python scripts/05_fleet.py                 # on data/processed/omie_prices.parquet
    uv run python scripts/05_fleet.py --regime normalization

Runs the battery dispatch against a price series that answers back, for a range
of fleet sizes, and reports revenue per MW, the resulting spread and the mean
price. Revenue per MW falling as the fleet grows is the effect the project is
about; read against the annualised cost of storage it gives the size at which
building more stops paying.

If the OMIE aggregated curves have been downloaded the price impact is read
from them. Otherwise it falls back to a single provisional slope, and says so:
any absolute figure from that fallback is not reportable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pce68.dispatch.battery import Battery
from pce68.dispatch.fleet import CurveImpact, LinearImpact, cannibalisation_curve
from pce68.features.regimes import annotate

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"

DUCK = np.array(
    [62, 56, 51, 48, 47, 52, 68, 84, 72, 44, 22, 11, 6, 12, 27, 48, 78, 112, 141, 152,
     146, 121, 96, 76], dtype=float
)


def demo_prices(days: int = 120) -> pd.DataFrame:
    """A season of duck-shaped days: solar trough at midday, evening peak."""
    rng = np.random.default_rng(68)
    start = pd.Timestamp("2024-04-01").tz_localize(TZ).tz_convert("UTC")
    shape = np.tile(DUCK, days) + rng.normal(0.0, 6.0, 24 * days)
    return pd.DataFrame(
        {
            "timestamp_utc": start + pd.to_timedelta(np.arange(24 * days), unit="h"),
            "price_es": shape,
            "resolution_min": 60,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="synthetic days instead of the parquet")
    parser.add_argument("--regime", default=None, help="keep only this regime, e.g. normalization")
    parser.add_argument("--output", default=None, help="where to write the CSV")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    bat = Battery(**config["battery"])
    fleet = config["fleet"]

    if args.demo:
        prices = demo_prices()
        label = "synthetic days"
    else:
        path = ROOT / config["data"]["processed_dir"] / "omie_prices.parquet"
        if not path.exists():
            raise SystemExit(f"{path} not found. Run `make data` first, or pass --demo.")
        prices = pd.read_parquet(path)
        label = "OMIE"

    if args.regime:
        prices = annotate(prices)
        prices = prices[prices["regime"].astype(str) == args.regime].reset_index(drop=True)
        if prices.empty:
            raise SystemExit(f"no periods in regime {args.regime!r}")
        label += f", regime {args.regime}"

    days = sorted(set(prices["timestamp_utc"].dt.tz_convert(TZ).dt.date))
    curves_dir = ROOT / fleet["curves_dir"]
    curve_files = sorted(curves_dir.glob("curva_pbc_*.1")) if curves_dir.exists() else []

    if curve_files:
        from pce68.data.omie_curves import load

        impact = CurveImpact(load(curves_dir, days))
        gaps = impact.missing(days, 24)
        log.info("Price impact from %d OMIE curve files; %d period-days without a curve.",
                 len(curve_files), gaps)
    else:
        slope = fleet["linear_slope_eur_mwh_per_gwh"]
        impact = LinearImpact(slope)
        log.warning(
            "No curve files in %s: falling back to a flat %.1f EUR/MWh per GWh.\n"
            "That slope is an assumption, not a measurement. The shape of the result is "
            "meaningful; the absolute figures are NOT reportable until the curves are in.",
            curves_dir, slope,
        )

    log.info("\n%d days (%s), battery of %g h at %.0f%% round trip\n",
             len(days), label, bat.duration_hours, 100 * bat.round_trip_efficiency)

    table = cannibalisation_curve(
        prices, bat, impact,
        fleet_sizes_mw=[float(x) for x in fleet["sizes_mw"]],
        tranche_mw=float(fleet["tranche_mw"]),
    )

    print(f"{'fleet':>9}  {'revenue':>14}  {'vs taker':>9}  {'spread':>9}  {'mean price':>11}")
    print(f"{'GW':>9}  {'EUR/MW/year':>14}  {'%':>9}  {'EUR/MWh':>9}  {'EUR/MWh':>11}")
    for row in table.itertuples():
        print(
            f"{row.fleet_mw / 1000:>9.2f}  {row.revenue_eur_per_mw_year:>14,.0f}  "
            f"{100 * row.revenue_vs_price_taker:>8.1f}%  {row.mean_spread_eur_mwh:>9.2f}  "
            f"{row.mean_price_eur_mwh:>11.2f}"
        )

    last = table.iloc[-1]
    print(
        f"\nAt {last.fleet_mw / 1000:.1f} GW the fleet keeps "
        f"{100 * last.revenue_vs_price_taker:.1f}% of what a price taker would earn, "
        f"and the {bat.duration_hours:g} h spread has moved from "
        f"{table['mean_spread_eur_mwh'].iloc[0]:.2f} to {last.mean_spread_eur_mwh:.2f} EUR/MWh."
    )

    destination = Path(args.output) if args.output else ROOT / "data" / "processed" / "fleet.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    log.info("\nWrote %s", destination)


if __name__ == "__main__":
    main()
