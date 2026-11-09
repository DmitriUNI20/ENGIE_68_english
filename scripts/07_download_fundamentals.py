"""Download the market fundamentals from REE's ESIOS: demand, wind, solar, ...

    python scripts/07_download_fundamentals.py                    # or `make fundamentals`
    python scripts/07_download_fundamentals.py --find "eolica"    # look up indicator ids

Needs the free ESIOS token in the ESIOS_TOKEN environment variable.

Downloads every indicator listed under `fundamentals.indicators` in
config/base.yaml, for the days of the impact panel (`impact.start` to
yesterday), and writes data/processed/fundamentals.parquet: one row per
quarter hour (UTC) and one column per indicator, plus residual demand
(demand minus wind and solar) when those three are present.

Each indicator in the config is an id or an official ESIOS name, which the
script looks up in the catalogue. The script prints the official name of
each one it downloads: if a name is not what its column says, look the right
id up with --find and put it in the config.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

from pce68.data.esios import find, indicator, resolve

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--find", metavar="TEXT", help="list ESIOS indicators whose name contains TEXT")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    with requests.Session() as s:
        if args.find:
            hits = find(s, args.find)
            print(
                hits.to_string(index=False)
                if not hits.empty
                else f"No indicator name contains {args.find!r}."
            )
            return

        cfg = config["fundamentals"]
        start = pd.Timestamp(config["impact"]["start"]).date()
        end = date.today() - timedelta(days=1)
        log.info("Downloading %d indicators from %s to %s ...\n", len(cfg["indicators"]), start, end)
        columns, failed = {}, []
        for column, ref in cfg["indicators"].items():
            indicator_id = ref
            try:
                indicator_id = resolve(s, ref)
                name, series = indicator(s, indicator_id, start, end)
            except Exception as e:  # noqa: BLE001 - report the indicator and carry on
                log.warning("  %-14s id %-6s FAILED: %s", column, indicator_id, e)
                failed.append(column)
                continue
            columns[column] = series
            log.info(
                "  %-14s id %-6s %-60s %d quarter hours",
                column,
                indicator_id,
                name[:60],
                series.notna().sum(),
            )

    if not columns:
        raise SystemExit("Nothing downloaded.")
    table = pd.DataFrame(columns)
    if {"demand", "wind", "solar_pv"} <= set(table.columns):
        table["residual_demand"] = table["demand"] - table["wind"] - table["solar_pv"]
    table = table.rename_axis("timestamp_utc").reset_index()

    output = ROOT / config["data"]["processed_dir"] / "fundamentals.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    log.info("\nWrote %s: %d quarter hours, columns %s", output, len(table), list(table.columns[1:]))
    log.info(
        "Check the names above: each must match its column. Fix wrong ids in config/base.yaml (use --find)."
    )
    if failed:
        log.warning("Failed: %s", failed)


if __name__ == "__main__":
    main()
