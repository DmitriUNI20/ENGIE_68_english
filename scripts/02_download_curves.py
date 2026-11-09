"""Download OMIE's aggregated curves and keep them in compact form.

    uv run python scripts/02_download_curves.py                # or `make curves`
    uv run python scripts/02_download_curves.py --sample 3     # 3 days a week only
    uv run python scripts/02_download_curves.py --start 2024-01-01
    uv run python scripts/02_download_curves.py --check        # validate, download nothing

A raw curva_pbc file is ~19 MB a day. Each one is parsed as it arrives and
reduced to data/processed/curves/curves_YYYYMMDD.parquet (a few hundred KB):
the effective curve plus each period's matched price, which is all the
simulation reads. The raw file is not kept unless `curves.keep_raw` is set.
Raw files already in `curves.raw_dir` (from an earlier download) are compacted
first, so nothing has to be fetched twice.

Downloads run `curves.workers` at a time. Days already compacted are skipped,
so the script can be stopped and resumed. `--sample N` keeps N days a week,
rotating the weekday from one week to the next so every weekday and season is
represented; the impact model does not need every day.

`--check` measures the reader before anything is built on it: it re-clears the
effective curves of every compacted day and compares the result with the
published Spanish price (omie_prices.parquet, from `make data`). Before
calibration the error should be ~1-2 EUR/MWh on quarter-hourly days; several
euros on most periods means the reader is wrong.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from pce68.data.omie_curves import (
    ENCODING,
    compact,
    compact_path,
    download_raw,
    effective_curves,
    file_name,
    parse_curves,
)
from pce68.market.curves import reconstruction_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"


def sampled(days: list[date], per_week: int) -> list[date]:
    """Keep `per_week` days a week, shifting the weekday each week."""
    return [d for d in days if (d.toordinal() + d.toordinal() // 7) % 7 < per_week]


def store(text: str, day: date, compact_dir: Path, raw_dir: Path, keep_raw: bool) -> None:
    compact(parse_curves(text)).to_parquet(compact_path(compact_dir, day), index=False)
    if keep_raw:
        (raw_dir / file_name(day)).write_text(text, encoding=ENCODING)


def published_by_day(prices: pd.DataFrame) -> dict[date, pd.DataFrame]:
    local = prices["timestamp_utc"].dt.tz_convert(TZ)
    ordered = prices.assign(day=local.dt.date).sort_values("timestamp_utc")
    return {d: g[["price_es", "price_pt"]] for d, g in ordered.groupby("day")}


def _summary(label: str, e: np.ndarray) -> None:
    if e.size == 0:
        return
    log.info(
        "  %-34s %7d %8.2f %7.0f%% %7.0f%% %9.2f",
        label, e.size, e.mean(), 100 * (e <= 1).mean(), 100 * (e <= 5).mean(), np.quantile(e, 0.99),
    )  # fmt: skip


def check(compact_dir: Path, price_file: Path) -> None:
    """Re-clear the effective curves of every compacted day against the published Spanish price.

    The error is broken down by file layout (hourly files before 01/10/2025 have
    no bid-type column, so rejected complex bids and the French flows cannot be
    told apart) and by whether Spain and Portugal cleared apart: the curves are
    Iberian, so on split periods the Spanish price does not come from one cross.
    """
    files = sorted(compact_dir.glob("curves_*.parquet"))
    if not files:
        raise SystemExit(f"no compact curves in {compact_dir}; download some first.")
    if not price_file.exists():
        raise SystemExit(f"{price_file} not found; run `make data` first.")
    published = published_by_day(pd.read_parquet(price_file))

    rows, skipped = [], 0
    for path in files:
        day = pd.Timestamp(path.stem.split("_")[-1]).date()
        compacted = pd.read_parquet(path)
        curves = effective_curves(compacted)
        prices = published.get(day)
        if prices is None or len(prices) != len(curves):
            skipped += 1
            continue
        err = np.abs(reconstruction_error(curves, prices["price_es"].to_numpy()))
        rows.append(
            pd.DataFrame(
                {
                    "day": day,
                    "quarter_hourly": bool((compacted["hours"] < 1).any()),
                    "split": (prices["price_es"] - prices["price_pt"]).abs().to_numpy() > 0.005,
                    "error": err,
                }
            )
        )
    if not rows:
        raise SystemExit("no day could be matched against a published price.")
    r = pd.concat(rows, ignore_index=True)

    log.info("Re-cleared %d periods of %d days against marginalpdbc (%d days skipped: no price or",
             len(r), r["day"].nunique(), skipped)  # fmt: skip
    log.info("a different number of periods, e.g. daylight-saving days)\n")
    log.info("  %-34s %7s %8s %8s %8s %9s", "", "periods", "MAE", "<=1 EUR", "<=5 EUR", "99% q.")
    _summary("all", r["error"].to_numpy())
    for qh, label in ((False, "hourly files (to 30/09/2025)"), (True, "quarter-hourly (from 01/10/2025)")):
        part = r[r["quarter_hourly"] == qh]
        _summary(label, part["error"].to_numpy())
        _summary("  Spain and Portugal coupled", part.loc[~part["split"], "error"].to_numpy())
        _summary("  Spain and Portugal split", part.loc[part["split"], "error"].to_numpy())

    worst = r.groupby("day")["error"].mean().sort_values(ascending=False).head(5)
    log.info("\n  Worst days (mean error, EUR/MWh): %s", ", ".join(f"{d} {v:.1f}" for d, v in worst.items()))
    log.info("\nThe simulation calibrates each period to the published price, so this residual")
    log.info("only affects how the curve bends near the price, not the starting point.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: curves.start)")
    parser.add_argument("--end", help="last day (default: curves.end, or yesterday)")
    parser.add_argument("--sample", type=int, choices=range(1, 8), help="days per week to keep")
    parser.add_argument("--check", action="store_true", help="validate the reader, download nothing")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    cfg = config["curves"]
    raw_dir, compact_dir = ROOT / cfg["raw_dir"], ROOT / cfg["compact_dir"]
    compact_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        check(compact_dir, ROOT / config["data"]["processed_dir"] / "omie_prices.parquet")
        return

    # Raw files from an earlier download: compact them instead of fetching again.
    raws = sorted(raw_dir.glob("curva_pbc_*.1")) if raw_dir.exists() else []
    if raws:
        log.info("Compacting %d curve files already in %s (about 1-2 s each) ...", len(raws), raw_dir)
    for n, raw in enumerate(raws, start=1):
        if n % 20 == 0 or n == len(raws):
            log.info("  file %d of %d", n, len(raws))
        day = pd.Timestamp(raw.name[10:18]).date()
        if not compact_path(compact_dir, day).exists():
            try:
                store(raw.read_text(encoding=ENCODING, errors="replace"), day, compact_dir, raw_dir, False)
            except ValueError as e:
                log.warning("%s: %s", raw.name, e)
                continue
        if not cfg.get("keep_raw", False):
            raw.unlink()

    start = pd.Timestamp(args.start or cfg["start"]).date()
    end_cfg = args.end or cfg.get("end")
    end = pd.Timestamp(end_cfg).date() if end_cfg else date.today() - timedelta(days=1)
    days = [d.date() for d in pd.date_range(start, end, freq="D")]
    if args.sample:
        days = sampled(days, args.sample)
    todo = [d for d in days if not compact_path(compact_dir, d).exists()]
    log.info("%d days wanted (%s to %s), %d already compact, %d to download with %d workers ...",
             len(days), start, end, len(days) - len(todo), len(todo), cfg["workers"])  # fmt: skip
    if cfg.get("keep_raw", False):
        raw_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    with requests.Session() as s, ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(download_raw, d, s): d for d in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                store(future.result(), day, compact_dir, raw_dir, cfg.get("keep_raw", False))
            except Exception as e:  # noqa: BLE001 - report the day and carry on
                log.warning("%s: %s", day, e)
                failed += 1
            if n % 20 == 0 or n == len(todo):
                log.info("  %d / %d days done", n, len(todo))

    log.info("Finished: %d days compact in %s, %d failed (run again to retry them).",
             len(days) - failed, compact_dir, failed)  # fmt: skip
    log.info("Next: validate the reader with  python scripts/02_download_curves.py --check")


if __name__ == "__main__":
    main()
