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


def test_revenue_check_prices_the_simulated_dispatch():
    from pce68.model.impact_model import error_by_fleet_size, revenue_check

    d = build_dataset(panel(), None)
    d["charge_mw"] = np.where(d["p0_rank"] <= 0.5, 1000.0 * d["bess_gw"], 0.0)
    d["discharge_mw"] = np.where(d["p0_rank"] > 0.5, 1000.0 * d["bess_gw"], 0.0)
    d["pred_perfect"] = d["delta"]
    d["pred_no impact"] = 0.0
    r = revenue_check(d, "perfect")
    assert np.allclose(r["model error %"], 0)
    e = error_by_fleet_size(d)
    assert np.allclose(e["perfect"], 0) and np.allclose(e.loc[2, "no impact"], 2)
