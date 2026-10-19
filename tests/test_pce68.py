from datetime import date

import numpy as np
import pandas as pd
import pytest

from pce68.data.omie import parse
from pce68.data.omie_curves import build as build_curves
from pce68.data.omie_curves import parse as parse_curves
from pce68.dispatch.battery import Battery, optimal_cycle, optimal_cycle_schedule, simulate_price_taker
from pce68.dispatch.fleet import LinearImpact, cannibalisation_curve, simulate_fleet
from pce68.features.metrics import daily_spread
from pce68.features.regimes import annotate, summary
from pce68.market.curves import Curve, clear, impact_curve, price_impact

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


# ── Aggregated curves


def ladder() -> tuple[Curve, Curve]:
    """Toy pair whose intersection can be checked by hand.

    Sell:  10 MWh at 20 EUR, 20 at 50, 30 at 90.
    Buy:   25 MWh at 20 EUR, 18 at 50,  5 at 90.
    Excess supply is -15 at 20, +2 at 50 and +25 at 90, so it clears at 50.
    """
    sell = Curve(np.array([20.0, 50.0, 90.0]), np.array([10.0, 20.0, 30.0]), "sell")
    buy = Curve(np.array([20.0, 50.0, 90.0]), np.array([25.0, 18.0, 5.0]), "buy")
    return buy, sell


def merit_order(scarcity: float = 1.0) -> tuple[Curve, Curve]:
    """A convex merit order: cheap steps packed together, dear ones far apart."""
    t = np.linspace(0.0, 1.0, 40)
    price = np.concatenate([[-5.0, 0.0], 30.0 + 160.0 * t**2])
    volume = np.cumsum(np.concatenate([[3000.0, 7000.0], np.full(40, 500.0)]))
    sell = Curve(price, volume, "sell")
    buy = Curve(np.array([60.0, 3000.0]), np.array([16000.0 * scarcity, 15000.0 * scarcity]), "buy")
    return buy, sell


def test_curve_steps_in_the_right_direction():
    buy, sell = ladder()
    assert sell.at(10.0) == 0.0 and sell.at(20.0) == 10.0 and sell.at(49.9) == 10.0
    assert buy.at(20.0) == 25.0 and buy.at(21.0) == 18.0 and buy.at(1000.0) == 0.0


def test_clearing_on_a_hand_checked_case():
    assert clear(*ladder()) == (50.0, 18.0)


def test_demand_above_supply_clears_at_the_cap():
    sell = Curve(np.array([10.0]), np.array([5.0]), "sell")
    buy = Curve(np.array([10.0]), np.array([100.0]), "buy")
    assert clear(buy, sell)[0] == 4000.0


def test_charging_lifts_the_price_and_discharging_depresses_it():
    buy, sell = merit_order()
    assert price_impact(buy, sell, 1000.0, "charge") >= 0.0
    assert price_impact(buy, sell, 1000.0, "discharge") <= 0.0
    assert price_impact(buy, sell, 0.0, "discharge") == 0.0


def test_discharging_more_never_raises_the_price():
    buy, sell = merit_order()
    impacts = impact_curve(buy, sell, np.array([0.0, 500.0, 1000.0, 2000.0, 4000.0]), "discharge")
    assert np.all(np.diff(impacts) <= 1e-9)


def test_price_impact_is_not_proportional_to_volume():
    """The reason a single slope cannot stand in for the curves."""
    buy, sell = merit_order()
    small, large = impact_curve(buy, sell, np.array([1000.0, 4000.0]), "discharge")
    assert not np.isclose(large / small, 4.0, rtol=0.05)


def test_shift_refuses_a_negative_volume():
    with pytest.raises(ValueError):
        ladder()[1].shift(-1.0)


CURVE_FILE = (
    "OMIE - Mercado de electricidad;Curvas agregadas;15/01/2024;\n"
    "Hora;Fecha;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);\n"
    "1;15/01/2024;V;1.000,5;0,00;O;\n"
    "1;15/01/2024;V;500,0;45,30;O;\n"
    "1;15/01/2024;C;2.000,0;180,00;O;\n"
    "1;15/01/2024;C;300,0;20,00;O;\n"
    "1;15/01/2024;V;900,0;45,30;C;\n"
)


def test_curve_reader_drops_matched_rows_and_reads_comma_decimals():
    bids = parse_curves(CURVE_FILE)
    assert len(bids) == 4  # the row flagged as matched is not part of the offered curve
    assert bids["energy_mwh"].max() == pytest.approx(2000.0)


def test_curve_reader_finds_columns_by_name_not_position():
    shuffled = CURVE_FILE.replace(
        "Hora;Fecha;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);",
        "Precio Compra/Venta;Hora;Fecha;Tipo Oferta;Energía Compra/Venta;Ofertada (O)/Casada (C);",
    ).replace("1;15/01/2024;V;1.000,5;0,00;O;", "0,00;1;15/01/2024;V;1.000,5;O;")
    assert parse_curves(shuffled)["price_eur_mwh"].iloc[0] == pytest.approx(0.0)


CURVE_HEADER_2025 = (
    "OMIE - Mercado de electricidad;Fecha Emisión :31/03/2025 - 16:59;;01/04/2025;Mercado diario;;;;\n"
    "Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;Precio Compra/Venta;"
    "Ofertada (O)/Casada (C);Tipología de Oferta;\n"
)


def curve_file_2025(periods: int) -> str:
    """A file in the layout OMIE switched to on 19/03/2025."""
    # Numbers in OMIE's own notation: dot for thousands, comma for decimals.
    rows = [
        f"{p};01/04/2025;MI;;{kind};{qty};{price};O;S;"
        for p in range(1, periods + 1)
        for kind, qty, price in (
            ("V", "1.000,0", "0,00"),
            ("V", "500,0", "45,30"),
            ("C", "2.000,0", "180,00"),
        )
    ]
    return CURVE_HEADER_2025 + "\n".join(rows) + "\n"


def test_reader_accepts_the_2025_layout():
    """Energia was renamed Potencia; the old reader aborted every file from 19/03/2025."""
    bids = parse_curves(curve_file_2025(periods=24))
    assert len(bids) == 72
    assert set(bids["kind"]) == {"C", "V"}


def test_tipologia_column_is_not_mistaken_for_tipo_oferta():
    """The 2025 files added 'Tipología de Oferta'; matching on 'tipo' alone would grab it."""
    bids = parse_curves(curve_file_2025(periods=1))
    assert sorted(bids["kind"].unique()) == ["C", "V"], "kind must come from 'Tipo Oferta', not 'S'"


def test_power_is_converted_to_energy_on_quarter_hourly_days():
    """1 MW over a quarter hour is 0.25 MWh. Without this the battery volume is 4x out."""
    hourly = parse_curves(curve_file_2025(periods=24))
    quarterly = parse_curves(curve_file_2025(periods=96))
    assert hourly["energy_mwh"].max() == pytest.approx(2000.0)
    assert quarterly["energy_mwh"].max() == pytest.approx(500.0)


def test_curves_accumulate_in_the_right_direction():
    buy, sell = build_curves(parse_curves(CURVE_FILE), period=1)
    assert sell.at(0.0) == pytest.approx(1000.5)
    assert sell.at(45.30) == pytest.approx(1500.5)
    assert buy.at(180.0) == pytest.approx(2000.0)
    assert buy.at(20.0) == pytest.approx(2300.0)


# ── Fleet


def duck(day: str, n_days: int = 4) -> pd.DataFrame:
    """A few days with a solar trough and an evening peak, the shape arbitrage lives on."""
    shape = np.array([60, 55, 50, 48, 48, 55, 70, 85, 70, 40, 20, 10, 5, 10, 25, 45, 75, 110, 140, 150,
                      145, 120, 95, 75], dtype=float)
    frames = [series(f"{day[:-2]}{int(day[-2:]) + i:02d}", (shape + i).tolist()) for i in range(n_days)]
    return pd.concat(frames, ignore_index=True)


def test_fleet_without_feedback_matches_the_price_taker():
    """The new machinery must be a strict generalisation of the tested engine."""
    prices, bat = duck("2024-05-01"), Battery(2, 0.85, 365)
    fleet = simulate_fleet(prices, bat, LinearImpact(0.0), fleet_mw=1.0).daily
    taker = simulate_price_taker(prices, bat)
    assert fleet["revenue_eur_per_mw"].to_numpy() == pytest.approx(taker["revenue_eur_per_mw"].to_numpy())
    assert fleet["cycles"].tolist() == taker["cycles"].tolist()


def test_schedule_agrees_with_the_margin_it_replaces():
    rng = np.random.default_rng(0)
    bat = Battery(2, 0.85, 365)
    for _ in range(20):
        p = rng.normal(80.0, 40.0, 24)
        h = np.ones(24)
        charge, discharge, margin = optimal_cycle_schedule(p, h, bat)
        assert margin == pytest.approx(optimal_cycle(p, h, bat))
        if margin > 0:
            assert (p * discharge).sum() - (p * charge).sum() == pytest.approx(margin)
            assert discharge.sum() == pytest.approx(bat.duration_hours)


def test_revenue_per_mw_falls_as_the_fleet_grows():
    """Self-cannibalisation: the feedback is what makes it appear at all."""
    prices, bat = duck("2024-05-01"), Battery(2, 0.85, 365)
    table = cannibalisation_curve(
        prices, bat, LinearImpact(8.0), fleet_sizes_mw=[500.0, 2000.0, 6000.0], tranche_mw=250.0
    )
    revenue = table["revenue_eur_per_mw_year"].to_numpy()
    assert np.all(np.diff(revenue) < 0), "every extra GW must cut what the fleet already there earns"
    assert table["mean_spread_eur_mwh"].is_monotonic_decreasing
    assert table["revenue_vs_price_taker"].iloc[0] == pytest.approx(1.0)


def test_a_fleet_large_enough_stops_entering():
    """Once the spread is arbitraged away no further tranche would be built."""
    prices, bat = duck("2024-05-01"), Battery(2, 0.85, 365)
    result = simulate_fleet(prices, bat, LinearImpact(60.0), fleet_mw=40000.0, tranche_mw=250.0)
    assert result.daily["discharged_mwh"].max() < 40000.0 * bat.duration_hours
