import numpy as np
import pandas as pd

from pce68.model.projection import adjust, load_forecast, predict_at, scenario_path, to_quarter_hours
from pce68.model.projection import test_forecast as make_test_forecast  # not a test itself


class Linear:
    """A model whose impact is -X €/MWh everywhere."""

    def predict(self, x):
        return -x["bess_gw"].to_numpy(dtype=float)


BUNDLE = {"model": Linear(), "features": ["bess_gw"]}


def test_scenario_path_is_linear_between_points_and_flat_outside():
    days = pd.Series(pd.to_datetime(["2025-06-01", "2027-01-01", "2027-07-02", "2031-05-05"]))
    gw = scenario_path({2027: 4, 2028: 8}, days)
    assert gw[0] == 4 and gw[1] == 4 and gw[3] == 8
    assert np.isclose(gw[2], 4 + 4 * 182 / 365)


def test_predictions_interpolate_between_simulated_sizes_and_cap_at_the_largest():
    x = pd.DataFrame({"p0": [50.0] * 4})
    delta = predict_at(BUNDLE, x, np.array([0.0, 3.0, 12.0, 30.0]), grid=[2, 4, 12, 20])
    assert np.allclose(delta, [0, -3, -12, -20])
    with_factors = {**BUNDLE, "factors": {2: 2.0, 4: 2.0}}
    assert np.allclose(predict_at(with_factors, x.iloc[:2], np.array([1.0, 3.0]), grid=[2, 4]), [-2, -6])


def test_adjusted_price_and_ratio():
    x = pd.DataFrame({"timestamp_utc": pd.date_range("2027-01-01", periods=3, freq="15min", tz="UTC"),
                      "p0": [100.0, 2.0, -10.0]})  # fmt: skip
    out = adjust(BUNDLE, x, np.array([10.0, 10.0, 10.0]), grid=[10], min_abs_price=5)
    assert np.allclose(out["price_adjusted"], [90, -8, -20])
    assert np.isclose(out["y_hat"][0], 0.9) and np.isnan(out["y_hat"][1]) and np.isclose(out["y_hat"][2], 2.0)


def test_hourly_forecast_is_spread_over_quarters(tmp_path):
    f = tmp_path / "forecast.csv"
    hourly = pd.DataFrame({"datetime": ["2027-03-01 00:00", "2027-03-01 01:00"], "Price": [40, 50]})
    hourly.to_csv(f, index=False)
    q = load_forecast(f)
    assert len(q) == 8 and q["price"].tolist() == [40] * 4 + [50] * 4
    assert q["timestamp_utc"].iloc[0] == pd.Timestamp("2027-02-28 23:00", tz="UTC")  # CET = UTC+1
    assert len(to_quarter_hours(q)) == 8  # already quarter-hourly: unchanged


def test_test_forecast_repeats_the_last_52_weeks_by_weekday():
    ts = pd.date_range("2025-11-01", "2026-11-01", freq="15min", tz="Europe/Madrid", inclusive="left")
    weekdays = pd.to_datetime(pd.Series(ts.date)).dt.weekday
    prices = pd.DataFrame({"timestamp_utc": ts.tz_convert("UTC"), "price_es": weekdays})
    f = make_test_forecast(prices, None, [2027, 2028])
    assert f["price"].notna().all() and len(f) == 4 * 24 * (365 + 366) - 0  # DST: 23 h and 25 h days cancel
    weekday = f["timestamp_utc"].dt.tz_convert("Europe/Madrid").dt.weekday
    assert (f["price"] == weekday).mean() > 0.999


def test_futures_forecast_keeps_the_shape_and_averages_each_future():
    from pce68.model.projection import futures_forecast

    ts = pd.date_range("2025-11-01", "2026-11-01", freq="15min", tz="Europe/Madrid", inclusive="left")
    shape = 50 + 30 * np.sin(np.arange(len(ts)) / 96 * 2 * np.pi)
    prices = pd.DataFrame({"timestamp_utc": ts.tz_convert("UTC"), "price_es": shape})
    f = futures_forecast(prices, None, {2027: 60.0, 2028: 75.0})
    year = f["timestamp_utc"].dt.tz_convert("Europe/Madrid").dt.year
    assert np.allclose(f.groupby(year)["price"].mean(), [60.0, 75.0])
    base = make_test_forecast(prices, None, [2027, 2028])
    ratio = f["price"] / base["price"]
    assert np.allclose(ratio[year == 2027], ratio[year == 2027].iloc[0])  # one factor per year: same shape
