"""Fit and validate the stage-2 impact model (roadmap step 2).

    python scripts/08_fit_impact_model.py        # or `make model`

Reads the impact panel (03_simulate_impact.py) and, if present, the ESIOS
fundamentals (07_download_fundamentals.py), and predicts every month with
models fitted on the others (leave one month out). Models compared:

    no impact                       the price does not move (benchmark)
    rank lookup, ridge              simple models
    gradient boosting               trees
    gradient boosting, weighted     trees, trading periods weigh more
    ... + calibrated                with a factor per fleet size so that the
                                    fleet's revenue matches on other months

Prints three tables: price error, price error by fleet size, and the error
of the fleet's revenue per MW and year by fleet size. The best model is the
one with the smallest mean revenue error over the fleet sizes, as the
cannibalisation curve is what stage 3 needs. It is fitted on all the data
and saved.

Writes:
    reports/impact_model_scores.csv     price error per model
    reports/impact_model_by_gw.csv      price error by fleet size
    reports/impact_model_revenue.csv    revenue per MW and year by fleet size and model
    data/processed/impact_model.joblib  the fitted best model, its features and size factors
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
    fit_model,
    model_names,
    revenue_by_model,
    revenue_errors,
    size_factors,
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

    data = build_dataset(panel, fundamentals, config["battery"]["duration_hours"])
    features = feature_columns(data)
    log.info("%d rows (%d days x periods x fleet sizes), %d months, features: %s",
             len(data), data["day"].nunique(), data["month_key"].nunique(), ", ".join(features))  # fmt: skip

    t0 = time.time()
    log.info("Leave-one-month-out validation (a few minutes) ...")
    predicted = cross_validate(data, features)
    log.info("Done in %.0f s. All on months the models did not see.\n", time.time() - t0)
    names = model_names(predicted)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    table = summarise(predicted)
    log.info("Error of the predicted price with storage:\n")
    print(table.round(3).to_string())
    table.to_csv(reports / "impact_model_scores.csv")

    by_gw = error_by_fleet_size(predicted)
    log.info("\nMAE (€/MWh) by fleet size:\n")
    print(by_gw[table.index].T.round(2).to_string())
    by_gw.to_csv(reports / "impact_model_by_gw.csv")

    revenue = revenue_by_model(predicted)
    errors = revenue_errors(revenue)
    ranking = errors.drop(columns="no impact").abs().mean().sort_values()
    log.info("\nRevenue per MW and year: simulated (EUR), then each model's error (%) against it:\n")
    simulated = revenue[["simulated"]].T.map("{:,.0f}".format)
    model_errors = errors.T.loc[[*ranking.index, "no impact"]].map("{:+.1f}".format)
    shown = pd.concat([simulated, model_errors])
    print(shown.rename_axis(None, axis=1).to_string())
    log.info("\nMean absolute revenue error over fleet sizes (%%):\n%s", ranking.round(2).to_string())
    revenue.to_csv(reports / "impact_model_revenue.csv")

    best = ranking.index[0]
    base = best.removesuffix(" + calibrated")
    clean = data.dropna(subset=[*features, "delta"])
    model = fit_model(base, clean, features)
    bundle = {"model": model, "features": features, "name": best}
    if best != base:
        bundle["factors"] = size_factors(predicted, predicted[f"pred_{base}"]).to_dict()
        log.info("\nSize factors: %s", {k: round(v, 3) for k, v in bundle["factors"].items()})
    out = processed / "impact_model.joblib"
    joblib.dump(bundle, out)
    log.info("\nBest: %s (of %d). Saved %s; tables in %s", best, len(names), out, reports)


if __name__ == "__main__":
    main()
