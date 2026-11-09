"""Stage 3: the BESS-adjusted price forecast, by storage scenario (roadmap step 3).

    python scripts/09_project.py        # or `make project`

For ENGIE's baseline forecast p^E and each storage scenario X_s of
config/base.yaml (`projection`), computes y_hat = f(t, X_s, z^E) with the model
saved by 08_fit_impact_model.py and the adjusted forecast p' = y_hat * p^E.
See src/pce68/model/projection.py for the details.

Until ENGIE's forecast is in `projection.forecast_file`, a TEST forecast is
used: the last 52 weeks of real prices (and ESIOS fundamentals) repeated for
each test year. Its results show how the method works, not a price outlook.

For each scenario and year it also reports the mean price, the mean daily
spread and the revenue of 1 MW of 2 h battery trading on the adjusted prices
(price taker, perfect foresight, one cycle a day): what one more MW would earn
in a market with X GW of storage. And it repeats the projection at each fixed
size of the simulated grid, which gives the cannibalisation curve on the
forecast, year by year.

Writes:
    reports/projection_prices.csv     quarter-hourly: base forecast and adjusted forecast per scenario
    reports/projection_summary.csv    per scenario and year
    reports/projection_curve.csv      revenue and spread per fixed size and year
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from pce68.dispatch.battery import Battery, simulate_price_taker
from pce68.model.impact_model import TZ, build_dataset, feature_columns, fit_model
from pce68.model.projection import adjust, forecast_features, load_forecast, scenario_path, test_forecast

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def model_for(bundle: dict, features: pd.DataFrame, config: dict, processed: Path) -> dict:
    """The saved model, or, if the forecast lacks some of its fundamentals, the same model
    refitted without them (keeping the saved size factors)."""
    inputs = [c for c in bundle["features"] if c != "bess_gw"]  # bess_gw is set per scenario
    missing = [c for c in inputs if c not in features or features[c].isna().all()]
    if not missing:
        return bundle
    log.warning("The forecast has no %s: refitting the model without them (about a minute).", missing)
    files = sorted((ROOT / config["impact"]["output_dir"]).glob("*.parquet"))
    panel = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    data = build_dataset(panel, None, config["battery"]["duration_hours"])
    names = [c for c in feature_columns(data) if c not in missing]
    base = bundle["name"].removesuffix(" + calibrated")
    model = fit_model(base, data.dropna(subset=[*names, "delta"]), names)
    return {**bundle, "model": model, "features": names}


def yearly(adjusted: pd.DataFrame, battery: Battery) -> pd.DataFrame:
    """Per year: GW, mean price, mean daily spread and 1 MW battery revenue, base and adjusted."""
    local = adjusted["timestamp_utc"].dt.tz_convert(TZ)
    a = adjusted.assign(year=local.dt.year, day=local.dt.date)
    days = a.groupby("year")["day"].nunique()
    out = pd.DataFrame({"GW (mean)": a.groupby("year")["bess_gw"].mean()})
    for label, col in (("base", "price_base"), ("adjusted", "price_adjusted")):
        out[f"mean price {label}"] = a.groupby("year")[col].mean()
        by_day = a.groupby(["year", "day"])[col]
        out[f"daily spread {label}"] = (by_day.max() - by_day.min()).groupby(level=0).mean()
        prices = pd.DataFrame({"timestamp_utc": a["timestamp_utc"], "price_es": a[col], "resolution_min": 15})
        revenue = simulate_price_taker(prices, battery)
        ryear = pd.to_datetime(revenue["day"]).dt.year
        out[f"revenue {label} (EUR/MW/yr)"] = revenue.groupby(ryear)["revenue_eur_per_mw"].sum() * 365 / days
    return out.rename_axis("year")


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    cfg = config["projection"]
    processed = ROOT / config["data"]["processed_dir"]
    model_file = processed / "impact_model.joblib"
    if not model_file.exists():
        raise SystemExit("No impact model; run scripts/08_fit_impact_model.py first.")
    bundle = joblib.load(model_file)
    # Sizes the model was fitted on (older bundles: the factors' keys, else the config grid).
    grid = bundle.get("grid") or list(bundle.get("factors", {})) or config["impact"]["grid_gw"]
    grid = sorted(float(g) for g in grid if g > 0)
    battery = Battery(**config["battery"])
    t0 = time.time()

    forecast_file = ROOT / cfg["forecast_file"]
    if forecast_file.exists():
        forecast = load_forecast(forecast_file)
        source = f"ENGIE forecast: {forecast_file.name}"
    else:
        prices = pd.read_parquet(processed / "omie_prices.parquet")
        fund_file = processed / "fundamentals.parquet"
        fundamentals = pd.read_parquet(fund_file) if fund_file.exists() else None
        forecast = test_forecast(prices, fundamentals, cfg["test_years"])
        source = "TEST forecast: last 52 weeks of real prices repeated each year. NOT a price outlook."
        log.warning("No %s: using a %s", cfg["forecast_file"], source)
    local = forecast["timestamp_utc"].dt.tz_convert(TZ)
    log.info("%s\n%d quarter hours, %s to %s", source, len(forecast), local.min().date(), local.max().date())

    features = forecast_features(forecast, config["battery"]["duration_hours"])
    bundle = model_for(bundle, features, config, processed)
    if features[bundle["features"][:-1]].isna().any(axis=1).any():
        log.warning("Some quarter hours lack inputs; they are filled from the previous one.")
        inputs = bundle["features"][:-1]  # all but bess_gw, set per scenario
        features[inputs] = features[inputs].ffill().bfill()
    days = features["timestamp_utc"].dt.tz_convert(TZ).dt.date

    # 1. Scenarios: the adjusted forecast along each storage path.
    prices_out = pd.DataFrame(
        {"timestamp_utc": features["timestamp_utc"], "datetime_local": local.dt.tz_localize(None)}
    )
    prices_out["base forecast"] = features["p0"].to_numpy()
    summaries = []
    for name, points in cfg["scenarios"].items():
        gw = scenario_path(points, days)
        if gw.max() > max(grid):
            log.warning("%s reaches %.0f GW, beyond the %g GW simulated: capped.", name, gw.max(), max(grid))
        adjusted = adjust(bundle, features, gw, grid, config["impact"]["min_abs_price"])
        prices_out[name] = adjusted["price_adjusted"].to_numpy()
        summaries.append(yearly(adjusted, battery).assign(scenario=name).reset_index())
    summary = pd.concat(summaries)[["scenario", "year", *summaries[0].columns.drop(["year", "scenario"])]]

    # 2. Fixed sizes: the cannibalisation curve on the forecast, year by year.
    curve = []
    for size in [0, *grid]:
        adjusted = adjust(bundle, features, pd.Series(float(size), index=features.index).to_numpy(), grid,
                          config["impact"]["min_abs_price"])  # fmt: skip
        y = yearly(adjusted, battery)
        curve.append(
            pd.DataFrame(
                {
                    "GW": size,
                    "year": y.index,
                    "revenue (EUR/MW/yr)": y["revenue adjusted (EUR/MW/yr)"].to_numpy(),
                    "daily spread": y["daily spread adjusted"].to_numpy(),
                    "mean price": y["mean price adjusted"].to_numpy(),
                }
            )
        )
    curve = pd.concat(curve, ignore_index=True)

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    prices_out["timestamp_utc"] = prices_out["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M")
    prices_out.to_csv(reports / "projection_prices.csv", index=False, float_format="%.2f")
    summary.to_csv(reports / "projection_summary.csv", index=False)
    curve.to_csv(reports / "projection_curve.csv", index=False)
    (reports / "projection_source.txt").write_text(source + "\n")

    log.info("\nDone in %.0f s. By scenario and year (prices and spreads in EUR/MWh):\n", time.time() - t0)
    print(summary.round(1).to_string(index=False))
    log.info("\nRevenue of 1 MW of 2 h battery (EUR/MW/yr) with X GW installed, by year:\n")
    table = curve.pivot(index="GW", columns="year", values="revenue (EUR/MW/yr)").round(0)
    print(table.rename(index="{:g}".format).to_string())
    log.info("\nWrote projection_prices.csv, projection_summary.csv and projection_curve.csv to %s", reports)


if __name__ == "__main__":
    main()
