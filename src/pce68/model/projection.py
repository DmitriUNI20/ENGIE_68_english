"""Stage 3 of the roadmap: apply the impact model to a price forecast.

Following the roadmap: let p^E(s, t) be ENGIE's baseline day-ahead forecast,
X_s the installed BESS capacity of a scenario on day s (for example a PNIEC
path) and z^E(s, t) the forecast's fundamentals. Then

    y_hat(s, t) = f(t, X_s, z^E(s, t))        p'(s, t) = y_hat(s, t) * p^E(s, t)

The model is fitted on the difference delta = p^X - p0 rather than on the
ratio (see impact_model.py), so the adjusted price is p' = p^E + delta_hat,
which is the same thing: y_hat = p' / p^E = 1 + delta_hat / p^E. The ratio is
also reported, left empty where |p^E| is close to zero.

X between two sizes of the simulated grid (say 13 GW, between 12 and 15)
interpolates linearly between the predictions at those two sizes, with no
impact at 0 GW. Beyond the largest simulated size the model is not used: X is
capped there and a warning is printed.

Repeating this for several X_s paths gives internally consistent price curves
for alternative storage scenarios without rerunning the curve simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pce68.model.impact_model import TZ, add_features, join_fundamentals, predict_delta

FUNDAMENTAL_COLUMNS = ["demand", "wind", "solar_pv", "hydro", "nuclear", "combined_cycle"]


# ── Forecast input


def to_quarter_hours(forecast: pd.DataFrame) -> pd.DataFrame:
    """Spread an hourly forecast over quarter hours (same value in each quarter); keep a
    quarter-hourly one as it is. The day-ahead market clears in quarter hours since 10/2025."""
    f = forecast.sort_values("timestamp_utc").reset_index(drop=True)
    step = f["timestamp_utc"].diff().median()
    if pd.isna(step) or step < pd.Timedelta(minutes=60):
        return f
    quarters = [f.assign(timestamp_utc=f["timestamp_utc"] + pd.Timedelta(minutes=15 * q)) for q in range(4)]
    return pd.concat(quarters).sort_values("timestamp_utc").reset_index(drop=True)


def load_forecast(path: Path) -> pd.DataFrame:
    """Read a price forecast from CSV, Excel or parquet.

    Columns: `price` (€/MWh) and either `timestamp_utc` (UTC) or `datetime`
    (Spanish local time; the start of each hour or quarter hour). Optional:
    `demand`, `wind`, `solar_pv`, `hydro`, `nuclear`, `combined_cycle` (MW),
    used by the model when present.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    elif path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "timestamp_utc" in raw:
        ts = pd.to_datetime(raw["timestamp_utc"], utc=True)
    elif "datetime" in raw:
        local = pd.to_datetime(raw["datetime"])
        ts = local.dt.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    else:
        raise ValueError(f"{path.name}: needs a `timestamp_utc` or `datetime` column")
    if "price" not in raw:
        raise ValueError(f"{path.name}: needs a `price` column (€/MWh)")
    out = pd.DataFrame({"timestamp_utc": ts, "price": pd.to_numeric(raw["price"], errors="coerce")})
    for c in FUNDAMENTAL_COLUMNS:
        if c in raw:
            out[c] = pd.to_numeric(raw[c], errors="coerce")
    return to_quarter_hours(out.dropna(subset=["timestamp_utc", "price"]))


def test_forecast(
    prices: pd.DataFrame, fundamentals: pd.DataFrame | None, years: list[int], last_day: str | None = None
) -> pd.DataFrame:
    """A stand-in forecast while ENGIE's is not available: the last 52 weeks of real prices
    (and fundamentals), repeated for each of `years`.

    Each future day takes the historical day 52, 104 ... weeks earlier that
    falls in the last 52 weeks, so weekdays and seasons line up (drift of
    about one day a year). Every year then has the same base prices, and any
    difference between years comes only from the storage path.
    """
    hist = prices[["timestamp_utc", "price_es"]].rename(columns={"price_es": "price"})
    if fundamentals is not None:
        columns = ["timestamp_utc", *[c for c in FUNDAMENTAL_COLUMNS if c in fundamentals]]
        hist = hist.merge(fundamentals[columns], on="timestamp_utc", how="left")
    local = hist["timestamp_utc"].dt.tz_convert(TZ)
    end = pd.Timestamp(last_day).date() if last_day else local.dt.date.max()
    hist = hist.assign(day=local.dt.date, clock=local.dt.strftime("%H:%M"))
    hist = hist[hist["day"] > end - pd.Timedelta(days=364)]
    by_clock = hist.drop(columns="timestamp_utc").groupby(["day", "clock"]).mean()

    first, after = f"{min(years)}-01-01", f"{max(years) + 1}-01-01"
    ts = pd.date_range(first, after, freq="15min", tz=TZ, inclusive="left")
    ts = ts[ts.year.isin(years)]
    target = pd.DataFrame({"timestamp_utc": ts.tz_convert("UTC")})
    days = pd.Series(ts.date)
    weeks_back = np.ceil((pd.to_datetime(days) - pd.Timestamp(end)).dt.days / 364).clip(lower=0)
    anchor = (pd.to_datetime(days) - pd.to_timedelta(weeks_back * 364, unit="D")).dt.date
    key = pd.MultiIndex.from_arrays([anchor, ts.strftime("%H:%M")])
    values = by_clock.reindex(key)
    out = pd.concat([target, values.reset_index(drop=True)], axis=1)
    # A quarter hour missing in the anchor day (daylight saving): carry the previous one.
    return out.ffill().bfill()


def futures_forecast(
    prices: pd.DataFrame, fundamentals: pd.DataFrame | None, futures: dict, last_day: str | None = None
) -> pd.DataFrame:
    """A public baseline: the shape of the last 52 weeks of real prices, rescaled each year to
    the OMIP baseload future for Spain of that year (EUR/MWh).

    Futures give the level the market expects for a whole year, not its hours;
    the historical quarter hours give the shape (solar trough, evening peak). Each
    year's prices are multiplied by future / mean of that year's base prices, so
    the year averages the future and its spreads scale with it. Fundamentals are
    the historical ones, unchanged.
    """
    years = sorted(int(y) for y in futures)
    out = test_forecast(prices, fundamentals, years, last_day)
    year = out["timestamp_utc"].dt.tz_convert(TZ).dt.year
    level = out.groupby(year)["price"].transform("mean")
    target = year.map({int(k): float(v) for k, v in futures.items()})
    out["price"] = out["price"] * target / level
    return out


# ── Scenarios


def scenario_path(points: dict, days: pd.Series) -> np.ndarray:
    """Installed GW on each day, linear between the dated points of a scenario.

    `points` maps a year (meaning 1 January) or a date to GW; before the
    first point the first value holds and after the last the last.
    """
    when = pd.to_datetime([f"{k}-01-01" if str(k).isdigit() else str(k) for k in points])
    gw = np.array(list(points.values()), dtype=float)
    order = np.argsort(when)
    x = when[order].to_numpy().astype("datetime64[D]").astype(float)
    t = pd.to_datetime(days).to_numpy().astype("datetime64[D]").astype(float)
    return np.interp(t, x, gw[order])


# ── Applying the model


def forecast_features(forecast: pd.DataFrame, duration_hours: float) -> pd.DataFrame:
    """The model's features for a quarter-hourly forecast (bess_gw is set later)."""
    local = forecast["timestamp_utc"].dt.tz_convert(TZ)
    d = forecast.assign(day=local.dt.date, hours=0.25, p0=forecast["price"])
    d = add_features(d, duration_hours, keys=["day"])
    fundamentals = forecast[["timestamp_utc", *[c for c in FUNDAMENTAL_COLUMNS if c in forecast]]]
    return join_fundamentals(d, fundamentals if fundamentals.shape[1] > 1 else None)


def predict_at(bundle: dict, features: pd.DataFrame, gw: np.ndarray, grid: list[float]) -> np.ndarray:
    """Predicted delta at any X: linear between the two neighbouring simulated sizes."""
    grid = sorted({0.0, *map(float, grid)})
    gw = np.clip(gw, 0, grid[-1])
    pos = np.searchsorted(grid, gw, side="right") - 1
    pos = np.clip(pos, 0, len(grid) - 2)
    lower, upper = np.array(grid)[pos], np.array(grid)[pos + 1]
    w = (gw - lower) / (upper - lower)
    delta = np.zeros(len(features))
    for size in grid[1:]:
        for side, weight in ((lower, 1 - w), (upper, w)):
            rows = (side == size) & (weight > 0)
            if rows.any():
                x = features.loc[rows].assign(bess_gw=size)
                delta[rows] += weight[rows] * predict_delta(bundle, x)
    return delta


def adjust(
    bundle: dict,
    features: pd.DataFrame,
    gw: np.ndarray,
    grid: list[float],
    min_abs_price: float,
) -> pd.DataFrame:
    """The BESS-adjusted forecast for one storage path: p' = p^E + delta_hat, and y_hat = p'/p^E."""
    delta = predict_at(bundle, features, gw, grid)
    base = features["p0"].to_numpy()
    adjusted = base + delta
    ratio = np.where(np.abs(base) >= min_abs_price, adjusted / np.where(base == 0, 1, base), np.nan)
    return pd.DataFrame(
        {
            "timestamp_utc": features["timestamp_utc"].to_numpy(),
            "bess_gw": np.clip(gw, 0, max(grid)),
            "price_base": base,
            "price_adjusted": adjusted,
            "y_hat": ratio,
        }
    )
