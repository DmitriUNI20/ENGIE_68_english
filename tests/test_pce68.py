from datetime import date

import numpy as np
import pandas as pd
import pytest

from pce68.data.omie import parse
from pce68.dispatch.battery import Battery, optimal_cycle, simulate_price_taker
from pce68.features.metrics import daily_spread
from pce68.features.regimes import annotate, summary

TZ = "Europe/Madrid"


def marginalpdbc(day: date, prices: list[float]) -> str:
    rows = [f"{day:%Y;%m;%d};{i};{p + 1:.2f};{p:.2f};" for i, p in enumerate(prices, start=1)]
    return "MARGINALPDBC;\n" + "\n".join(rows) + "\n*\n"


def series(day: str, prices: list[float], resolution: int = 60) -> pd.DataFrame:
    t0 = pd.Timestamp(day).tz_localize(TZ).tz_convert("UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": t0 + pd.to_timedelta(np.arange(len(prices)) * resolution, unit="min"),
            "price_es": np.asarray(prices, float),
            "resolution_min": resolution,
        }
    )


# ── OMIE


@pytest.mark.parametrize(
    ("day", "n", "resolution"),
    [
        (date(2024, 1, 15), 24, 60),
        (date(2024, 3, 31), 23, 60),  # switch to summer time
        (date(2024, 10, 27), 25, 60),  # switch to winter time
        (date(2025, 10, 15), 96, 15),  # quarter-hourly day-ahead
        (date(2025, 10, 26), 100, 15),
    ],
)
def test_parse_covers_the_local_day(day, n, resolution):
    df = parse(marginalpdbc(day, list(range(n))))
    assert len(df) == n and (df["resolution_min"] == resolution).all()
    local = df["timestamp_utc"].dt.tz_convert(TZ)
    assert local.iloc[0] == pd.Timestamp(day).tz_localize(TZ)
    assert (local.dt.date == day).all()
    assert df["timestamp_utc"].diff().dropna().eq(pd.Timedelta(minutes=resolution)).all()
    # Spain is the last column, Portugal the one before
    assert df["price_es"].tolist() == [float(i) for i in range(n)]
    assert df["price_pt"].tolist() == [float(i + 1) for i in range(n)]


def test_parse_rejects_periods_that_do_not_fit():
    with pytest.raises(ValueError):
        parse(marginalpdbc(date(2024, 1, 15), [50.0] * 23))


# ── Spread


def test_hourly_spread():
    p = [10, 0, 50, 40, 100, 90] + [60] * 18
    s = daily_spread(series("2024-05-01", p), 2.0)
    assert s.columns.tolist() == ["day_local", "spread_2h"]
    assert s["spread_2h"].iloc[0] == pytest.approx((100 + 90) / 2 - (0 + 10) / 2)


def test_spread_same_in_quarter_hours_as_in_hours():
    p = np.r_[np.full(8, 20.0), np.full(8, 80.0), np.full(8, 50.0)]
    h = daily_spread(series("2024-05-01", p.tolist()), 2.0)["spread_2h"].iloc[0]
    q = daily_spread(series("2024-05-01", np.repeat(p, 4).tolist(), 15), 2.0)["spread_2h"].iloc[0]
    assert h == pytest.approx(q) == pytest.approx(60.0)


# ── Battery


def test_cycle_respects_time_order():
    # Expensive comes before cheap: with no discharging before charging there is no margin.
    bat = Battery(1, 1.0, 365)
    p = np.array([100.0, 0.0])
    assert optimal_cycle(p, np.ones(2), bat) == 0.0
    assert optimal_cycle(p[::-1].copy(), np.ones(2), bat) == pytest.approx(100.0)


def test_cycle_with_efficiency():
    bat = Battery(2, 0.8, 365)
    # Selling 2 MWh at 100 requires buying 2.5 MWh: 2 h at 0 and half an hour at 10.
    p = np.array([0.0, 0.0, 10.0, 10.0, 100.0, 100.0])
    assert optimal_cycle(p, np.ones(6), bat) == pytest.approx(200 - 0.5 * 10)


def test_simulation_applies_the_cycle_cap():
    days = pd.concat(
        [series(f"2024-05-0{i}", [0.0] * 12 + [10.0 * i] * 12) for i in range(1, 4)], ignore_index=True
    )
    r = simulate_price_taker(days, Battery(2, 1.0, 2))
    assert r["cycles"].tolist() == [0, 1, 1]
    assert r["revenue_eur_per_mw"].tolist() == pytest.approx([0, 40, 60])


# ── Regimes


def test_regimes_by_local_date():
    # 30/06/2021 22:30 UTC is already 01/07 in Madrid: gas_crisis
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2021-06-30 21:30", "2021-06-30 22:30", "2022-06-15 00:00", "2024-01-01 00:00"], utc=True
            ),
            "price_es": [1.0, 2.0, 3.0, 4.0],
            "resolution_min": 60,
        }
    )
    assert annotate(df)["regime"].astype(str).tolist() == [
        "pre_crisis",
        "gas_crisis",
        "iberian_exception",
        "normalization",
    ]
    r = summary(df)
    assert r["regime"].tolist() == ["pre_crisis", "gas_crisis", "iberian_exception", "normalization"]
    assert r.columns.tolist() == ["regime", "periods", "start", "end", "mean_price", "max_price"]


def test_download_falls_back_to_yearly_zip(tmp_path, monkeypatch):
    import io
    import zipfile

    import requests

    from pce68.data import omie

    days = [date(2021, 1, 1), date(2021, 1, 2)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for d in days:
            z.writestr(f"marginalpdbc_2021/{omie.file_name(d)}", marginalpdbc(d, [50.0] * 24))
    requested = []

    class Resp:
        def __init__(self, status, content=b""):
            self.status_code, self.content = status, content
            self.text = content.decode("latin-1")

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code}")

    def get(self, url, timeout):
        requested.append(url)
        return Resp(200, buf.getvalue()) if url.endswith("2021.zip") else Resp(404)

    monkeypatch.setattr(requests.Session, "get", get)
    assert omie.download(days, tmp_path, pause_s=0) == []
    assert sorted(f.name for f in tmp_path.iterdir()) == [omie.file_name(d) for d in days]
    assert sum(u.endswith(".zip") for u in requested) == 1  # the zip is requested only once
    assert len(omie.consolidate(tmp_path)) == 48
