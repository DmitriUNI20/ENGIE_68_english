"""Stage 2 of the roadmap: a fast approximation of the market-impact simulator.

The simulator (stage 1) answers "what would the price of this quarter hour
have been with X GW of storage?" from the day's OMIE curves. Curves do not
exist for future days, so a statistical model learns that answer from the
historical panel and can then be applied to any price forecast:

    delta(t, X) = p^X - p0 = f(t, X, z) + error

The roadmap states the target as a ratio y = p^X / p0. The model is fitted on
the difference instead: Spain has hundreds of hours a year at or near zero,
where the ratio explodes, while the difference stays bounded. The ratio follows
from it, p^X / p0 = 1 + delta / p0, wherever p0 is not near zero.

Features (z): calendar (hour, quarter, weekday, month), the original price and
its rank within the day (storage charges in the cheapest periods and
discharges in the dearest, so the rank drives the sign of delta), the day's
price level and spread, and the fundamentals from ESIOS (demand, wind, solar,
residual demand, renewable share). No day identifier, as the roadmap asks.

Validation leaves one calendar month out at a time: the model is scored on a
month it has not seen, so seasonal effects cannot leak from training to test.

Two refinements aim at the fleet's revenue, which is what the cannibalisation
curve needs:

* weights: periods where the fleet charges or discharges weigh more in the
  fit, as only they enter its revenue;
* calibration by fleet size: regression shrinks the largest impacts towards
  the mean, so the fitted prices are too kind to storage. For each fleet size
  a factor rescales the predicted delta so that the revenue of the simulated
  dispatch matches on the training months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TZ = "Europe/Madrid"

BASE_FEATURES = [
    "hour",
    "quarter",
    "weekday",
    "weekend",
    "month_sin",
    "month_cos",
    "p0",
    "p0_rank",
    "day_mean",
    "day_spread",
    "day_top_2h",
    "day_bottom_2h",
]
FUNDAMENTALS = ["demand", "wind", "solar_pv", "residual_demand", "renewable_share"]


def _mean_of_extremes(prices: pd.Series, hours: pd.Series, duration: float, top: bool) -> float:
    n = max(1, round(duration / hours.iloc[0]))
    return float((prices.nlargest(n) if top else prices.nsmallest(n)).mean())


def add_features(d: pd.DataFrame, duration_hours: float, keys: list[str]) -> pd.DataFrame:
    """Calendar and daily price-shape features. Needs timestamp_utc, hours, p0 and the `keys`
    that identify one day's prices (day, plus bess_gw in the panel, where days repeat per X)."""
    d = d.copy()
    local = d["timestamp_utc"].dt.tz_convert(TZ)
    d["hour"] = local.dt.hour
    d["quarter"] = local.dt.minute // 15
    d["weekday"] = local.dt.weekday
    d["weekend"] = (d["weekday"] >= 5).astype(int)
    month = local.dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    d["month_key"] = local.dt.strftime("%Y-%m")

    by_day = d.groupby(keys)["p0"]
    d["p0_rank"] = by_day.rank(pct=True)
    d["day_mean"] = by_day.transform("mean")
    d["day_spread"] = by_day.transform("max") - by_day.transform("min")
    # Mean price of the dearest and cheapest `duration_hours` of the day: where the fleet trades.
    for name, top in (("day_top_2h", True), ("day_bottom_2h", False)):
        per_day = d.groupby(keys)[["p0", "hours"]].apply(
            lambda g, top=top: _mean_of_extremes(g["p0"], g["hours"], duration_hours, top)
        )
        idx = d.set_index(keys).index
        d[name] = idx.map(per_day).to_numpy()
    return d


def join_fundamentals(d: pd.DataFrame, fundamentals: pd.DataFrame | None) -> pd.DataFrame:
    """Join demand, wind, solar ... by quarter hour (UTC) and add the renewable share."""
    if fundamentals is None:
        return d
    f = fundamentals.copy()
    if {"wind", "solar_pv", "demand"} <= set(f.columns):
        f["renewable_share"] = (f["wind"] + f["solar_pv"]) / f["demand"]
    return d.drop(columns=[c for c in f.columns if c != "timestamp_utc" and c in d]).merge(
        f, on="timestamp_utc", how="left"
    )


def build_dataset(
    panel: pd.DataFrame, fundamentals: pd.DataFrame | None, duration_hours: float = 2.0
) -> pd.DataFrame:
    """One row per (day, period, X > 0) with features and the target `delta`.

    Periods where Spain and Portugal cleared apart are dropped (the Iberian
    curves do not yield the Spanish price there), and so is the X = 0 row,
    which is the reference itself.
    """
    d = panel[panel["bess_gw"] > 0].copy()
    if "es_pt_split" in d:
        d = d[d["es_pt_split"].fillna(False).astype(bool) == False]  # noqa: E712 - NA-safe

    local_midnight = pd.to_datetime(d["day"]).dt.tz_localize(TZ)
    d["timestamp_utc"] = local_midnight.dt.tz_convert("UTC") + pd.to_timedelta(
        (d["period"] - 1) * d["hours"], unit="h"
    )
    d = add_features(d, duration_hours, keys=["day", "bess_gw"])

    # Weight in the fit: 1, plus up to 4 more for the share of the fleet's power traded.
    if {"charge_mw", "discharge_mw"} <= set(d.columns):
        traded = (d["discharge_mw"] - d["charge_mw"]).abs() / (d["bess_gw"] * 1000)
        d["weight"] = 1 + 4 * traded.clip(upper=1)
    else:
        d["weight"] = 1.0

    d = join_fundamentals(d, fundamentals)
    return d.reset_index(drop=True)


def feature_columns(data: pd.DataFrame) -> list[str]:
    return BASE_FEATURES + [c for c in FUNDAMENTALS if c in data.columns] + ["bess_gw"]


class RidgeWithStorage:
    """Ridge regression where every feature also enters multiplied by X and X².

    The impact is zero at X = 0 and grows with X, so a plain linear term in
    each feature would miss most of it; the products let each feature say how
    much it amplifies the effect of storage.
    """

    def __init__(self, alpha: float = 1.0):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    @staticmethod
    def _design(x: pd.DataFrame) -> np.ndarray:
        gw = x["bess_gw"].to_numpy()[:, None]
        other = x.drop(columns="bess_gw").to_numpy()
        return np.hstack([other * gw, other * gw**2, gw, gw**2])

    def fit(self, x: pd.DataFrame, y: np.ndarray, sample_weight=None) -> RidgeWithStorage:
        self.model.fit(self._design(x), y, ridge__sample_weight=sample_weight)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._design(x))


class RankLookup:
    """Benchmark: mean delta by decile of the price's rank within the day and by X."""

    def fit(self, x: pd.DataFrame, y: np.ndarray, sample_weight=None) -> RankLookup:
        key = pd.DataFrame({"decile": (x["p0_rank"] * 10).clip(upper=9.999).astype(int), "gw": x["bess_gw"]})
        self.table = pd.Series(y, index=pd.MultiIndex.from_frame(key)).groupby(level=[0, 1]).mean()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([(x["p0_rank"] * 10).clip(upper=9.999).astype(int), x["bess_gw"]])
        return self.table.reindex(key).fillna(0.0).to_numpy()


def _boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)


def models() -> dict[str, object]:
    return {
        "rank lookup": RankLookup(),
        "ridge": RidgeWithStorage(alpha=1.0),
        "gradient boosting": _boosting(),
        "gradient boosting, weighted": _boosting(),
    }


WEIGHTED = {"gradient boosting, weighted"}
CALIBRATED = ["gradient boosting", "gradient boosting, weighted"]


def fit_model(name: str, data: pd.DataFrame, features: list[str]):
    model = models()[name]
    weight = data["weight"].to_numpy() if name in WEIGHTED else None
    return model.fit(data[features], data["delta"].to_numpy(), sample_weight=weight)


def _traded_energy(d: pd.DataFrame) -> pd.Series:
    return (d["discharge_mw"] - d["charge_mw"]) * d["hours"]


def size_factors(d: pd.DataFrame, pred: pd.Series) -> pd.Series:
    """Per fleet size, the factor on predicted delta that makes the dispatch's revenue match.

    Revenue = sum(energy * (p0 + delta)), so matching it means
    sum(energy * delta) = a * sum(energy * predicted delta).
    """
    e = _traded_energy(d)
    num = (e * d["delta"]).groupby(d["bess_gw"]).sum()
    den = (e * pred).groupby(d["bess_gw"]).sum()
    return (num / den).where(den.abs() > 0, 1.0)


def cross_validate(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Leave-one-month-out predictions of delta for each model and the no-impact benchmark.

    Returns the rows used, with one column `pred_<model>` of predicted delta
    per model, all made on months the model did not see. The calibrated
    variants get their factors from the other months' predictions: the
    test month never sets its own factor.
    """
    data = data.dropna(subset=[*features, "delta"])
    predicted = []
    for month in sorted(data["month_key"].unique()):
        train, test = data[data["month_key"] != month], data[data["month_key"] == month]
        preds = {"no impact": np.zeros(len(test))}
        for name in models():
            preds[name] = fit_model(name, train, features).predict(test[features])
        predicted.append(test.assign(**{f"pred_{name}": p for name, p in preds.items()}))
    out = pd.concat(predicted)
    if {"charge_mw", "discharge_mw"} <= set(out.columns):
        for name in CALIBRATED:
            col = out[f"pred_{name}"]
            calibrated = pd.Series(np.nan, index=out.index)
            for month in out["month_key"].unique():
                other = out["month_key"] != month
                factors = size_factors(out[other], col[other])
                rows = ~other
                calibrated[rows] = col[rows] * out.loc[rows, "bess_gw"].map(factors).fillna(1.0)
            out[f"pred_{name} + calibrated"] = calibrated
    return out


def model_names(predicted: pd.DataFrame) -> list[str]:
    return [c.removeprefix("pred_") for c in predicted.columns if c.startswith("pred_")]


def summarise(predicted: pd.DataFrame) -> pd.DataFrame:
    """MAE of the predicted price with storage (€/MWh), overall and in each day's cheapest and
    dearest 10 %, and R² (against each month's own mean), from out-of-month predictions."""
    delta = predicted["delta"]
    low, high = predicted["p0_rank"] <= 0.1, predicted["p0_rank"] > 0.9
    tot = ((delta - delta.groupby(predicted["month_key"]).transform("mean")) ** 2).sum()
    rows = {}
    for name in model_names(predicted):
        err = predicted[f"pred_{name}"] - delta
        rows[name] = {
            "MAE (€/MWh)": err.abs().mean(),
            "MAE cheapest 10%": err[low].abs().mean(),
            "MAE dearest 10%": err[high].abs().mean(),
            "R²": 1 - (err**2).sum() / tot,
        }
    return pd.DataFrame(rows).T.rename_axis("model").sort_values("MAE (€/MWh)")


def error_by_fleet_size(predicted: pd.DataFrame) -> pd.DataFrame:
    """MAE (€/MWh) of each model for each fleet size, on unseen months: one column per model."""
    err = pd.DataFrame(
        {n: (predicted[f"pred_{n}"] - predicted["delta"]).abs() for n in model_names(predicted)},
        index=predicted.index,
    )
    return err.groupby(predicted["bess_gw"]).mean().rename_axis("GW")


def revenue_by_model(predicted: pd.DataFrame) -> pd.DataFrame:
    """Fleet revenue per MW and year (EUR) by fleet size: the simulated dispatch priced at the
    simulated prices and at each model's. "no impact" prices it at the original prices, i.e.
    ignoring cannibalisation. Spain-Portugal splits are left out, so the simulated column can
    differ slightly from fleet.csv."""
    energy = _traded_energy(predicted)
    revenue = pd.DataFrame(
        {"simulated": energy * predicted["p_x"]}
        | {n: energy * (predicted["p0"] + predicted[f"pred_{n}"]) for n in model_names(predicted)}
    )
    table = revenue.groupby(predicted["bess_gw"]).sum()
    days = predicted["day"].nunique()
    return (table.div(table.index.to_series() * 1000, axis=0) * 365 / days).rename_axis("GW")


def revenue_errors(revenue: pd.DataFrame) -> pd.DataFrame:
    """Error (%) of each model's revenue against the simulated one, by fleet size."""
    return 100 * (revenue.drop(columns="simulated").div(revenue["simulated"], axis=0) - 1)


def predict_delta(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    """Predicted delta from a saved bundle (model, features, optional size factors)."""
    pred = bundle["model"].predict(x[bundle["features"]])
    factors = bundle.get("factors")
    if factors is not None:
        pred = pred * x["bess_gw"].map(factors).fillna(1.0).to_numpy()
    return pred
