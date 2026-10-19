"""Download the OMIE aggregated supply and demand curves.

    uv run python scripts/02_download_curves.py              # or `make curves`
    uv run python scripts/02_download_curves.py --check      # validate the reader

These are the files the price impact is measured from, and the reason the
project does not need to assume a slope. They are public with roughly a ninety
-day delay and the archive is a rolling six years, so the early part of the
price series is already at the edge of what can still be fetched.

One day of curves is far heavier than one day of prices, so the default span is
the normalization regime only -- the market being projected. Widen it with
--start if a longer counterfactual is wanted.

`--check` runs the validation that decides whether the reader can be trusted:
it re-clears the untouched historical curves and compares the result against
the published marginalpdbc price. Gottfried and Muller report a mean error of
0.05 EUR/MWh in 2024 from ignoring block bids and market coupling. An error of
that order means the reader works; an error of several euros means it does not,
and every figure downstream is then wrong.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pce68.data.omie import consolidate
from pce68.data.omie_curves import build, download, file_name, parse
from pce68.market.curves import clear

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"


def check(curves_dir: Path, raw_prices: Path) -> None:
    """Re-clear untouched curves and compare with the published price."""
    files = sorted(curves_dir.glob("curva_pbc_*.1"))
    if not files:
        raise SystemExit(f"no curve files in {curves_dir}; download some first.")

    published = consolidate(raw_prices)
    published = published.assign(
        day=published["timestamp_utc"].dt.tz_convert(TZ).dt.date,
        period=published.groupby(published["timestamp_utc"].dt.tz_convert(TZ).dt.date).cumcount() + 1,
    ).set_index(["day", "period"])["price_es"]

    errors = []
    for path in files:
        stamp = path.stem.split("_")[-1]
        day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        bids = parse(path.read_text(encoding="latin-1", errors="replace"))
        for period in sorted(bids["period"].unique()):
            key = (day, int(period))
            if key not in published.index:
                continue
            try:
                price, _ = clear(*build(bids, int(period)))
            except ValueError:
                continue
            errors.append(price - float(published.loc[key]))

    if not errors:
        raise SystemExit("no period could be matched against a published price.")

    e = np.abs(np.array(errors))
    log.info("\nRe-cleared %d periods against marginalpdbc", len(e))
    log.info("  mean absolute error   %8.3f EUR/MWh   (Germany 2024: 0.05)", e.mean())
    log.info("  99%% quantile          %8.3f EUR/MWh   (Germany 2024: 1.06)", np.quantile(e, 0.99))
    log.info("  worst                 %8.3f EUR/MWh", e.max())
    verdict = "the reader looks sound" if e.mean() < 0.5 else "TOO LARGE -- do not trust what follows"
    log.info("  verdict: %s", verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01", help="first day, AAAA-MM-DD")
    parser.add_argument("--end", default=None, help="last day; default is 100 days ago")
    parser.add_argument("--check", action="store_true", help="validate the reader, download nothing")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    curves_dir = ROOT / config["fleet"]["curves_dir"]

    if args.check:
        check(curves_dir, ROOT / config["data"]["raw_dir"])
        return

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else date.today() - timedelta(days=100)
    days = [d.date() for d in pd.date_range(start, end, freq="D")]

    log.info("Downloading %d days of curves (%s to %s) into %s ...", len(days), start, end, curves_dir)
    failed = download(days, curves_dir)
    have = sum(1 for d in days if (curves_dir / file_name(d)).exists())
    log.info("%d of %d days present.", have, len(days))
    if failed:
        log.warning(
            "%d days failed. Days more recent than the ~90-day delay are not published yet; "
            "run again later for those.", len(failed)
        )
    log.info("\nNow validate the reader:  uv run python scripts/02_download_curves.py --check")


if __name__ == "__main__":
    main()
