"""Fit and validate the stage-2 impact model (roadmap step 2).

    python scripts/08_fit_impact_model.py        # or `make model`

Reads the impact panel (03_simulate_impact.py) and, if present, the ESIOS
fundamentals (07_download_fundamentals.py). Scores each model on months it
has not seen (leave one month out), prints the comparison, fits the best one
on all the data and saves it for stage 3.

Then two checks of the best model, also on unseen months:
    error by fleet size     MAE of each model for 1, 2, ... 20 GW
    revenue check           revenue per MW and year from the model's prices
                            against the simulated ones: does the model
                            reproduce the cannibalisation curve?

Writes:
    reports/impact_model_scores.csv     per-model scores
    reports/impact_model_by_gw.csv      error by fleet size
    reports/impact_model_revenue.csv    revenue check
    data/processed/impact_model.joblib  the fitted best model and its features
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from pce68.model.impact_model import (
    build_dataset,
    cross_validate,
    error_by_fleet_size,
    feature_columns,
    models,
    revenue_check,
    summarise,
)

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
    scores, predicted = cross_validate(data, features)
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

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "impact_model_scores.csv")
    log.info("\nBest: %s. Saved %s", best, out)

    by_gw = error_by_fleet_size(predicted)
    log.info("\nMAE (€/MWh) by fleet size, unseen months:\n")
    print(by_gw[table.index].round(2).to_string())
    by_gw.to_csv(reports / "impact_model_by_gw.csv")

    revenue = revenue_check(predicted, best)
    log.info("\nRevenue per MW and year (EUR), same dispatch priced three ways, unseen months (%s):\n", best)
    print(
        revenue.to_string(
            formatters={
                "simulated": "{:,.0f}".format,
                "model": "{:,.0f}".format,
                "no impact": "{:,.0f}".format,
                "model error %": "{:+.1f}".format,
                "no impact error %": "{:+.1f}".format,
            }
        )
    )
    revenue.to_csv(reports / "impact_model_revenue.csv")
    log.info("\nWrote the three tables to %s", reports)


if __name__ == "__main__":
    main()
