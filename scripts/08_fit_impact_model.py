"""Fit and validate the stage-2 impact model (roadmap step 2).

    python scripts/08_fit_impact_model.py        # or `make model`

Reads the impact panel (03_simulate_impact.py) and, if present, the ESIOS
fundamentals (07_download_fundamentals.py). Scores each model on months it
has not seen (leave one month out), prints the comparison, fits the best one
on all the data and saves it for stage 3.

Writes:
    reports/impact_model_scores.csv     per-model scores
    data/processed/impact_model.joblib  the fitted best model and its features
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from pce68.model.impact_model import build_dataset, cross_validate, feature_columns, models, summarise

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    processed = ROOT / config["data"]["processed_dir"]
    files = sorted((ROOT / config["impact"]["output_dir"]).glob("*.parquet"))
    if not files:
        raise SystemExit("No impact panel; run scripts/03_simulate_impact.py first.")
    panel = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    fund_file = processed / "fundamentals.parquet"
    fundamentals = pd.read_parquet(fund_file) if fund_file.exists() else None
    if fundamentals is None:
        log.info(
            "No %s: fitting without demand, wind and solar (run 07_download_fundamentals.py).", fund_file.name
        )

    data = build_dataset(panel, fundamentals)
    features = feature_columns(data)
    log.info("%d rows (%d days x periods x fleet sizes), %d months, features: %s",
             len(data), data["day"].nunique(), data["month_key"].nunique(), ", ".join(features))  # fmt: skip

    t0 = time.time()
    log.info("Leave-one-month-out validation (a few minutes) ...")
    scores = cross_validate(data, features)
    table = summarise(scores)
    log.info(
        "Done in %.0f s. Error of the predicted price with storage, on unseen months:\n", time.time() - t0
    )
    print(table.round(3).to_string())

    best = table.index[0]
    if best == "no impact":
        raise SystemExit("No model beats 'no impact'; nothing saved.")
    clean = data.dropna(subset=[*features, "delta"])
    model = models()[best].fit(clean[features], clean["delta"].to_numpy())
    out = processed / "impact_model.joblib"
    joblib.dump({"model": model, "features": features, "name": best}, out)

    report = ROOT / "reports" / "impact_model_scores.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(report)
    log.info("\nBest: %s. Saved %s and %s", best, out, report)


if __name__ == "__main__":
    main()
