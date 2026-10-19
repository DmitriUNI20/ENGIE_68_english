"""Download the OMIE day-ahead prices and store them as parquet.

    uv run python scripts/01_download_omie.py     # or `make data`

Only fetches the days not already in data/raw/omie, so re-running it is
cheap. It then consolidates everything into data/processed/omie_prices.parquet,
which is what the analysis and the models read.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from pce68.data.omie import consolidate, download

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    data = config["data"]
    start = pd.Timestamp(data["start"]).date()
    end = pd.Timestamp(data["end"]).date() if data.get("end") else date.today() - timedelta(days=1)
    days = [d.date() for d in pd.date_range(start, end, freq="D")]

    raw = ROOT / data["raw_dir"]
    log.info("Downloading %d days (%s to %s) into %s ...", len(days), start, end, raw)
    failed = download(days, raw)
    if failed:
        log.warning("%d days failed; run the script again to retry them.", len(failed))

    prices = consolidate(raw)
    output = ROOT / data["processed_dir"] / "omie_prices.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output, index=False)
    log.info(
        "Wrote %s: %d periods, %s to %s",
        output,
        len(prices),
        prices["timestamp_utc"].min(),
        prices["timestamp_utc"].max(),
    )


if __name__ == "__main__":
    main()
