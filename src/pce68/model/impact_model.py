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
]
FUNDAMENTALS = ["demand", "wind", "solar_pv", "residual_demand", "renewable_share"]


def build_dataset(panel: pd.DataFrame, fundamentals: pd.DataFrame | None) -> pd.DataFrame:
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
    local = d["timestamp_utc"].dt.tz_convert(TZ)
    d["hour"] = local.dt.hour
    d["quarter"] = local.dt.minute // 15
    d["weekday"] = local.dt.weekday
    d["weekend"] = (d["weekday"] >= 5).astype(int)
    month = local.dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    d["month_key"] = local.dt.strftime("%Y-%m")

    by_day = d.groupby(["day", "bess_gw"])["p0"]
    d["p0_rank"] = by_day.rank(pct=True)
    d["day_mean"] = by_day.transform("mean")
    d["day_spread"] = by_day.transform("max") - by_day.transform("min")

    if fundamentals is not None:
        f = fundamentals.copy()
        if {"wind", "solar_pv", "demand"} <= set(f.columns):
            f["renewable_share"] = (f["wind"] + f["solar_pv"]) / f["demand"]
        d = d.merge(f, on="timestamp_utc", how="left")
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

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> RidgeWithStorage:
        self.model.fit(self._design(x), y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._design(x))


class RankLookup:
    """Benchmark: mean delta by decile of the price's rank within the day and by X."""

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> RankLookup:
        key = pd.DataFrame({"decile": (x["p0_rank"] * 10).clip(upper=9.999).astype(int), "gw": x["bess_gw"]})
        self.table = pd.Series(y, index=pd.MultiIndex.from_frame(key)).groupby(level=[0, 1]).mean()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([(x["p0_rank"] * 10).clip(upper=9.999).astype(int), x["bess_gw"]])
        return self.table.reindex(key).fillna(0.0).to_numpy()


def models() -> dict[str, object]:
    return {
        "rank lookup": RankLookup(),
        "ridge": RidgeWithStorage(alpha=1.0),
        "gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0),
    }


def cross_validate(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-month-out scores for each model, plus the no-impact benchmark.

    MAE is of the price the model predicts (p0 + predicted delta) against the
    simulated p^X, in €/MWh, overall and in the cheapest and dearest decile of
    each day, where storage charges and discharges.

    Also returns the out-of-month predictions: the rows used, with one column
    `pred_<model>` of predicted delta per model, for the checks below.
    """
    data = data.dropna(subset=[*features, "delta"])
    rows, predicted = [], []
    for month in sorted(data["month_key"].unique()):
        train, test = data[data["month_key"] != month], data[data["month_key"] == month]
        preds = {"no impact": np.zeros(len(test))}
        for name, model in models().items():
            model.fit(train[features], train["delta"].to_numpy())
            preds[name] = model.predict(test[features])
        predicted.append(test.assign(**{f"pred_{name}": p for name, p in preds.items()}))
        for name, p in preds.items():
            err = np.abs(p - test["delta"].to_numpy())
            rows.append(
                {
                    "month": month,
                    "model": name,
                    "n": len(test),
                    "abs_err_sum": err.sum(),
                    "low_err_sum": err[test["p0_rank"].to_numpy() <= 0.1].sum(),
                    "low_n": int((test["p0_rank"] <= 0.1).sum()),
                    "high_err_sum": err[test["p0_rank"].to_numpy() > 0.9].sum(),
                    "high_n": int((test["p0_rank"] > 0.9).sum()),
                    "sq_err_sum": ((p - test["delta"].to_numpy()) ** 2).sum(),
                    "sq_tot_sum": ((test["delta"] - test["delta"].mean()) ** 2).sum(),
                }
            )
    return pd.DataFrame(rows), pd.concat(predicted)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby("model").sum(numeric_only=True)
    out = pd.DataFrame(
        {
            "MAE (€/MWh)": g["abs_err_sum"] / g["n"],
            "MAE cheapest 10%": g["low_err_sum"] / g["low_n"],
            "MAE dearest 10%": g["high_err_sum"] / g["high_n"],
            "R²": 1 - g["sq_err_sum"] / g["sq_tot_sum"],
        }
    )
    return out.sort_values("MAE (€/MWh)")


def error_by_fleet_size(predicted: pd.DataFrame) -> pd.DataFrame:
    """MAE (€/MWh) of each model for each fleet size, on unseen months: one column per model."""
    names = [c.removeprefix("pred_") for c in predicted.columns if c.startswith("pred_")]
    err = pd.DataFrame(
        {n: (predicted[f"pred_{n}"] - predicted["delta"]).abs() for n in names}, index=predicted.index
    )
    return err.groupby(predicted["bess_gw"]).mean().rename_axis("GW")


def revenue_check(predicted: pd.DataFrame, model: str) -> pd.DataFrame:
    """Fleet revenue per MW and year from the simulated prices and from the model's, by fleet size.

    Both use the simulated dispatch of the fleet, so the difference is only the
    error of the predicted prices, weighted by the energy traded in each
    period: what matters for the cannibalisation curve. "No impact" prices the
    same dispatch at the original prices, i.e. ignoring cannibalisation.
    Rows are those of the validation (Spain-Portugal splits are left out), so
    the simulated column can differ slightly from fleet.csv.
    """
    energy = (predicted["discharge_mw"] - predicted["charge_mw"]) * predicted["hours"]
    revenue = pd.DataFrame(
        {
            "simulated": energy * predicted["p_x"],
            "model": energy * (predicted["p0"] + predicted[f"pred_{model}"]),
            "no impact": energy * predicted["p0"],
        }
    )
    gw = predicted["bess_gw"]
    table = revenue.groupby(gw).sum()
    days = predicted["day"].nunique()
    table = table.div(table.index.to_series() * 1000, axis=0) * 365 / days
    table["model error %"] = 100 * (table["model"] / table["simulated"] - 1)
    table["no impact error %"] = 100 * (table["no impact"] / table["simulated"] - 1)
    return table.rename_axis("GW")
