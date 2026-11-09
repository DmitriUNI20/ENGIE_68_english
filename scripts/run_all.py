"""Run the whole pipeline, in order, and end with the Excel report.

    python scripts/run_all.py              # or `make all`
    python scripts/run_all.py --sample 3   # curves for 3 days a week only (faster first run)

Every step is incremental: prices and curves already downloaded and days
already simulated are skipped, so after the first run this only fetches and
simulates what is new. The Excel is written last so that it includes the
Cannibalisation sheet built from the steps before it.

    1. 01_download_omie.py     prices          data/processed/omie_prices.parquet
    2. 02_download_curves.py   OMIE curves     data/processed/curves/
    3. 03_simulate_impact.py   X GW of storage data/processed/impact/
    4. 05_fleet.py             fleet table     data/processed/fleet.csv
    5. 06_plot_fleet.py        chart           reports/cannibalisation_curve.png
    6. 04_export_excel.py      Excel report    reports/exploratory_analysis.xlsx

Stops at the first step that fails, and says which one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Prices", "01_download_omie.py"),
    ("OMIE curves", "02_download_curves.py"),
    ("Storage impact", "03_simulate_impact.py"),
    ("Fleet table", "05_fleet.py"),
    ("Chart", "06_plot_fleet.py"),
    ("Excel report", "04_export_excel.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, choices=range(1, 8), help="curve days per week to download")
    args = parser.parse_args()

    start = time.time()
    for n, (name, script) in enumerate(STEPS, start=1):
        extra = ["--sample", str(args.sample)] if args.sample and script.startswith("02_") else []
        print(f"\n=== {n}/{len(STEPS)}  {name}  ({script}) ===", flush=True)
        t = time.time()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStopped: step {n} ({script}) failed. Fix it and run again; finished steps are not redone.")
        print(f"--- done in {time.time() - t:.0f} s", flush=True)

    print(f"\nAll done in {(time.time() - start) / 60:.1f} min.")
    print(f"  Excel: {ROOT / 'reports' / 'exploratory_analysis.xlsx'}")
    print(f"  Chart: {ROOT / 'reports' / 'cannibalisation_curve.png'}")


if __name__ == "__main__":
    main()
