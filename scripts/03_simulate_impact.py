"""Simulate the price impact of X GW of storage on every downloaded day (roadmap stage 1).

    uv run python scripts/03_simulate_impact.py        # or `make impact`

Reads the compact curves left by 02_download_curves.py and, for each day,
calibrates them to the historical price and dispatches a fleet for every X
on `impact.grid_gw`. Writes data/processed/impact/YYYYMMDD.parquet: one row
per (period, X) with p0, pX, delta = pX - p0 and ratio = pX / p0. Days already
written are skipped, so it can be stopped and resumed. Days before
`impact.start` are left out (see the config for why).

`es_pt_split` marks periods where Spain and Portugal cleared at different
prices. The curves are Iberian, so there the Spanish price does not come from
a single cross and the simulated impact is less reliable; the impact model
can drop or down-weight them.

The historical price is the published Spanish one from omie_prices.parquet
(`make data`) when its periods match the curves; otherwise the matched
Iberian price stored with the curves.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from pce68.market.impact import simulate_day

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"


def published_prices(prices: pd.DataFrame | None, day: date, periods: int) -> pd.DataFrame | None:
    """Published Spanish and Portuguese prices of `day`, if the price file has exactly `periods` of them."""
    if prices is None:
        return None
    day_prices = prices.loc[prices["day"] == day, ["price_es", "price_pt"]]
    return day_prices.reset_index(drop=True) if len(day_prices) == periods else None


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    cfg, bat = config["impact"], config["battery"]
    compact_dir = ROOT / config["curves"]["compact_dir"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(cfg["start"]).date()
    files = [
        f for f in sorted(compact_dir.glob("curves_*.parquet")) if pd.Timestamp(f.stem[7:]).date() >= start
    ]
    if not files:
        raise SystemExit(f"no curves from {start} in {compact_dir}; run scripts/02_download_curves.py first.")

    price_file = ROOT / config["data"]["processed_dir"] / "omie_prices.parquet"
    prices = None
    if price_file.exists():
        prices = pd.read_parquet(price_file).sort_values("timestamp_utc")
        prices["day"] = prices["timestamp_utc"].dt.tz_convert(TZ).dt.date
    else:
        log.info("No %s: calibrating to the matched price stored with the curves.", price_file.name)

    log.info("%d days of curves; fleets %s GW, %s h, efficiency %s",
             len(files), cfg["grid_gw"], bat["duration_hours"], bat["round_trip_efficiency"])  # fmt: skip
    done = failed = 0
    t0 = time.time()
    for path in files:
        day = pd.Timestamp(path.stem.split("_")[-1]).date()
        output = out_dir / f"{day:%Y%m%d}.parquet"
        if output.exists():
            continue
        try:
            curves = pd.read_parquet(path)
            published = published_prices(prices, day, curves["period"].nunique())
            result = simulate_day(
                curves,
                cfg["grid_gw"],
                bat["duration_hours"],
                bat["round_trip_efficiency"],
                base_prices=None if published is None else published["price_es"].to_numpy(),
                min_abs_price=cfg["min_abs_price"],
            )
            if published is None:
                result["es_pt_split"] = pd.NA
            else:
                split = ((published["price_es"] - published["price_pt"]).abs() > 0.005).to_numpy()
                result["es_pt_split"] = split[result["period"].to_numpy() - 1]
        except Exception as e:  # noqa: BLE001 - report the day and carry on
            log.warning("%s: %s", day, e)
            failed += 1
            continue
        result.insert(0, "day", pd.Timestamp(day))
        result.to_parquet(output, index=False)
        done += 1
        if done % 20 == 0:
            log.info("  %d days simulated (%.0f s)", done, time.time() - t0)

    log.info("Finished: %d days simulated, %d failed. Results in %s", done, failed, out_dir)
    log.info("Next: the cannibalisation table with  python scripts/05_fleet.py")


if __name__ == "__main__":
    main()
