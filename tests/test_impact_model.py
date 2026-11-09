import numpy as np
import pandas as pd

from pce68.model.impact_model import RankLookup, build_dataset, feature_columns


def panel(day="2026-03-10", gws=(0, 1, 2)) -> pd.DataFrame:
    p0 = np.linspace(10, 105, 96)
    rows = []
    for gw in gws:
        rows.append(
            pd.DataFrame(
                {
                    "day": pd.Timestamp(day),
                    "period": np.arange(1, 97),
                    "hours": 0.25,
                    "bess_gw": gw,
                    "p0": p0,
                    "p_x": p0 - gw,
                    "es_pt_split": [True] + [False] * 95,
                }
            ).assign(delta=lambda d: d["p_x"] - d["p0"])
        )
    return pd.concat(rows, ignore_index=True)


def test_dataset_drops_the_reference_and_split_periods():
    d = build_dataset(panel(), None)
    assert set(d["bess_gw"]) == {1, 2}  # X = 0 is the reference itself
    assert len(d) == 2 * 95  # period 1 is a Spain-Portugal split


def test_dataset_times_and_ranks():
    d = build_dataset(panel(), None)
    first = d[d["bess_gw"] == 1].iloc[0]  # period 2 = 00:15 local on 10/03/2026 (UTC+1)
    assert first["timestamp_utc"] == pd.Timestamp("2026-03-09 23:15", tz="UTC")
    assert (first["hour"], first["quarter"], first["weekday"]) == (0, 1, 1)
    assert d["p0_rank"].between(0, 1).all()
    assert d.loc[d["p0"].idxmax(), "p0_rank"] == 1.0


def test_fundamentals_are_joined_by_quarter_hour():
    t = pd.date_range("2026-03-09 23:00", periods=96, freq="15min", tz="UTC")
    f = pd.DataFrame({"timestamp_utc": t, "demand": 30000.0, "wind": 6000.0, "solar_pv": 3000.0})
    d = build_dataset(panel(), f)
    assert d["demand"].notna().all() and d["renewable_share"].iloc[0] == 0.3
    assert "renewable_share" in feature_columns(d) and "bess_gw" in feature_columns(d)


def test_rank_lookup_learns_mean_delta_by_decile_and_size():
    d = build_dataset(panel(), None)
    model = RankLookup().fit(d, d["delta"].to_numpy())
    assert np.allclose(model.predict(d), d["delta"])  # delta is -X everywhere here


def traded(d: pd.DataFrame) -> pd.DataFrame:
    """Charge in the cheapest half of the day, discharge in the dearest, at full power."""
    full, dear = 1000.0 * d["bess_gw"], d["p0"] > d["p0"].median()
    return d.assign(charge_mw=np.where(dear, 0.0, full), discharge_mw=np.where(dear, full, 0.0))


def test_extremes_and_weights():
    d = build_dataset(traded(panel()), None)
    prices = np.linspace(10, 105, 96)[1:]  # period 1 is a split, dropped
    assert np.isclose(d["day_top_2h"].iloc[0], prices[-8:].mean())  # 2 h = 8 quarter hours
    assert np.isclose(d["day_bottom_2h"].iloc[0], prices[:8].mean())
    assert set(d["weight"]) == {5.0}  # full power traded in every period here
    assert set(build_dataset(panel(), None)["weight"]) == {1.0}  # no dispatch columns: equal weights


def test_size_factors_rescale_to_the_simulated_revenue():
    from pce68.model.impact_model import predict_delta, size_factors

    d = build_dataset(traded(panel()), None)
    half = d["delta"] / 2  # a model that predicts only half of each impact
    factors = size_factors(d, half)
    assert np.allclose(factors, 2.0)

    class Half:
        def predict(self, x):
            return x["bess_gw"].to_numpy() * -0.5

    bundle = {"model": Half(), "features": ["bess_gw"], "factors": factors.to_dict()}
    assert np.allclose(predict_delta(bundle, d), d["delta"])


def test_revenue_and_errors_by_model():
    from pce68.model.impact_model import error_by_fleet_size, revenue_by_model, revenue_errors

    d = build_dataset(traded(panel()), None)
    d = d.assign(**{"pred_perfect": d["delta"], "pred_no impact": 0.0})
    errors = revenue_errors(revenue_by_model(d))
    assert np.allclose(errors["perfect"], 0)
    e = error_by_fleet_size(d)
    assert np.allclose(e["perfect"], 0) and np.allclose(e.loc[2, "no impact"], 2)


def test_neighbouring_prices_stay_within_the_day():
    d = build_dataset(panel(), None)
    one = d[d["bess_gw"] == 1].sort_values("period").reset_index(drop=True)
    assert np.allclose(one["p0_prev_1h"].iloc[4:], one["p0"].shift(4).iloc[4:])  # 4 quarter hours back
    assert one["p0_prev_1h"].iloc[0] == one["p0"].iloc[0]  # day edge: own price
    assert one["p0_next_1h"].iloc[-1] == one["p0"].iloc[-1]


def test_gam_is_zero_without_storage_and_learns_a_linear_impact():
    from pce68.model.impact_model import GamWithStorage

    d = build_dataset(panel(gws=(0, 1, 2, 4)), None)
    x = d[["hour", "p0", "p0_rank", "bess_gw"]]
    gam = GamWithStorage(alpha=0.01).fit(x, d["delta"].to_numpy())
    assert np.abs(gam.predict(x) - d["delta"]).mean() < 0.05  # delta = -X here
    assert np.abs(gam.predict(x.assign(bess_gw=0.0))).max() < 0.01  # no storage, (almost) no impact


def test_importance_and_conditions():
    from pce68.model.impact_model import impact_by_condition, importance

    class SignOfRank:
        def predict(self, x):
            return np.where(x["p0_rank"] > 0.5, -x["bess_gw"], x["bess_gw"]).astype(float)

    d = build_dataset(traded(panel(gws=(0, 1, 2))), None)
    d["delta"] = SignOfRank().predict(d)
    bundle = {"model": SignOfRank(), "features": ["hour", "p0_rank", "bess_gw"]}
    w = importance(bundle, d, n=1000)
    assert w.index[0] in ("p0_rank", "bess_gw") and w["hour"] == 0
    c = impact_by_condition(d.assign(pred_m=d["delta"]), "m", gw=2)
    dear = c[(c["dimension"] == "hour") & (c["side"] == "dearest 10%")]
    assert (dear["simulated"] == -2).all() and (dear["model"] == -2).all()
