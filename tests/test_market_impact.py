import numpy as np
import pytest

from pce68.dispatch.fleet import Fleet, dispatch_fleet, fleet_revenue
from pce68.market.curves import PRICE_CAP, Curves, clear, reconstruction_error

# Supply stack used throughout: 1000 MW at 0 €, 1000 MW at 50 €, 1000 MW at 100 €.
SUPPLY = ([0.0, 50.0, 100.0], [1000.0, 1000.0, 1000.0])


def period(demand_mw: float) -> Curves:
    """Inelastic demand (bid at the cap) against the standard supply stack."""
    return Curves(SUPPLY[0], SUPPLY[1], [PRICE_CAP], [demand_mw])


# ── Clearing


def test_clear_finds_the_marginal_step():
    assert clear(period(500)) == 0.0
    assert clear(period(1500)) == 50.0
    assert clear(period(2500)) == 100.0


def test_extra_demand_raises_and_extra_supply_lowers_the_price():
    c = period(1500)
    assert clear(c, extra_demand_mw=600) == 100.0  # 2100 MW needed: third step
    assert clear(c, extra_supply_mw=600) == 0.0  # only 900 MW left for the stack


def test_scarcity_clears_at_the_cap():
    assert clear(period(5000)) == PRICE_CAP


def test_demand_step_can_set_the_price():
    # Demand: 2000 MW up to 60 €, 500 MW up to 200 €. At 50 € the stack offers
    # 2000 MW against 2500 MW wanted; above 60 € only 500 MW is wanted. The
    # 60 € demand bid is partly served and sets the price.
    c = Curves(SUPPLY[0], SUPPLY[1], [60.0, 200.0], [2000.0, 500.0])
    assert clear(c) == 60.0


def test_supply_step_sets_the_price_when_demand_would_pay_more():
    # Demand bids 1500 MW up to 60 €; the 50 € supply step covers it.
    c = Curves(SUPPLY[0], SUPPLY[1], [60.0], [1500.0])
    assert clear(c) == 50.0


def test_reconstruction_error_is_zero_when_curves_match():
    err = reconstruction_error([period(500), period(1500)], np.array([0.0, 50.0]))
    assert np.allclose(err, 0)


# ── Fleet dispatch


def test_no_fleet_leaves_prices_unchanged():
    day = dispatch_fleet([period(500), period(2100)], np.ones(2), Fleet(0, 2, 0.85))
    assert day["price"].tolist() == day["base_price"].tolist() == [0.0, 100.0]


def test_small_fleet_shaves_the_peak():
    # 300 MW charged at 0 € keeps the trough at 0 €; 300 MW discharged at the peak
    # leaves 1800 MW for the stack, which now clears at 50 €.
    hours = np.ones(2)
    day = dispatch_fleet([period(500), period(2100)], hours, Fleet(300, 1, 1.0))
    assert day["charge_mw"].tolist() == pytest.approx([300, 0])
    assert day["discharge_mw"].tolist() == pytest.approx([0, 300])
    assert day["price"].tolist() == [0.0, 50.0]
    assert fleet_revenue(day, hours) == pytest.approx(300 * 50)


def test_fleet_never_discharges_before_charging():
    # Expensive first, cheap later: no profitable cycle exists.
    day = dispatch_fleet([period(2100), period(500)], np.ones(2), Fleet(500, 1, 1.0))
    assert day["charge_mw"].sum() == day["discharge_mw"].sum() == 0


def test_large_fleet_stops_when_the_spread_no_longer_pays():
    # A huge fleet would flatten the day; it must stop before the peak falls
    # below the trough and never trade at a loss.
    hours = np.ones(2)
    fleet = Fleet(5000, 1, 0.8)
    day = dispatch_fleet([period(500), period(2100)], hours, fleet)
    assert day["discharge_mw"].sum() < fleet.power_mw
    assert day["price"].iloc[1] >= day["price"].iloc[0]
    assert fleet_revenue(day, hours) >= 0


def test_power_limit_and_energy_are_respected_in_quarter_hours():
    curves = [period(500)] * 8 + [period(2100)] * 8  # 2 h trough, 2 h peak
    hours = np.full(16, 0.25)
    fleet = Fleet(200, 1, 1.0)
    day = dispatch_fleet(curves, hours, fleet)
    assert (day["charge_mw"] <= fleet.power_mw + 1e-6).all()
    assert (day["discharge_mw"] <= fleet.power_mw + 1e-6).all()
    assert (day["discharge_mw"] * hours).sum() == pytest.approx(fleet.power_mw * fleet.duration_hours)


def test_large_fleet_equilibrium_by_hand():
    # Blocks of 100 MWh sold / 125 MWh bought. Four blocks keep the trough at
    # 0 € (500 MW spare at 0 €) and pull the peak to 50 €. A fifth would lift
    # the trough to 50 €, and 50 · 0.8 < 50: it is rejected.
    hours = np.ones(2)
    day = dispatch_fleet([period(500), period(2100)], hours, Fleet(5000, 1, 0.8))
    assert day["charge_mw"].iloc[0] == pytest.approx(500)
    assert day["discharge_mw"].iloc[1] == pytest.approx(400)
    assert day["price"].tolist() == [0.0, 50.0]
    assert fleet_revenue(day, hours) == pytest.approx(400 * 50)


def test_more_storage_narrows_the_daily_spread():
    # A duck-shaped day: low demand at midday (solar), high in the evening.
    demand = [1600, 1500, 1400, 1400, 1500, 1800, 2100, 2200, 1500, 900, 600, 500,
              500, 600, 900, 1400, 2000, 2500, 2800, 2900, 2700, 2300, 1900, 1700]  # fmt: skip
    stack = ([-10, 0, 20, 40, 60, 80, 100, 150, 250], [400] * 9)
    curves = [Curves(stack[0], stack[1], [PRICE_CAP], [d]) for d in demand]
    hours = np.ones(24)
    spreads = []
    for gw in [0, 0.5, 1, 2, 4]:
        p = dispatch_fleet(curves, hours, Fleet(gw * 1000, 2, 0.85))["price"]
        spreads.append(p.max() - p.min())
    assert spreads[0] > spreads[-1]
    assert all(a >= b for a, b in zip(spreads, spreads[1:], strict=False))


# ── Calibration


@pytest.mark.parametrize("target", [0.0, 50.0, 100.0])
def test_calibration_reproduces_the_target_price(target):
    from pce68.market.curves import calibrate, shifted

    c = period(1500)  # clears at 50 € on its own
    assert clear(shifted(c, calibrate(c, target))) == target


def test_calibrated_day_with_no_fleet_returns_the_historical_prices():
    from pce68.market.curves import calibrated

    curves = calibrated([period(1500), period(1500)], np.array([0.0, 100.0]))
    day = dispatch_fleet(curves, np.ones(2), Fleet(0, 2, 0.85))
    assert day["price"].tolist() == [0.0, 100.0]


# ── OMIE curve files

HEADER = (
    "OMIE - Mercado de electricidad;Fecha Emisión :01/11/2026 - 13:32;;02/11/2026;Mercado diario;;;;\n\n"
    "Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;Precio Compra/Venta;"
    "Ofertada (O)/Casada (C);Tipología de Oferta;\n"
)
SAMPLE = (
    HEADER
    + """H1Q1;02/11/2026;MI;;C;1.500,0;1.500,00;O;S;
H1Q1;02/11/2026;MI;;V;1.000,0;0,00;O;S;
H1Q1;02/11/2026;MI;;V;1.000,0;50,00;O;S;
H1Q1;02/11/2026;MI;;V;800,0;10,00;O;C02;
H1Q1;02/11/2026;MI;;V;300,0;10,00;C;C02;
H1Q1;02/11/2026;MI;;C;200,0;1.500,00;C;Exp FR;
H1Q1;02/11/2026;MI;;V;1.300,0;50,00;C;S;
H1Q2;02/11/2026;MI;;C;500,0;1.500,00;O;S;
H1Q2;02/11/2026;MI;;V;1.000,0;0,00;O;S;
H1Q2;02/11/2026;MI;;V;500,0;0,00;C;S;
;;;;;;;;;
"""
)


def test_parse_curves_reads_the_omie_format():
    from pce68.data.omie_curves import compact, parse_curves, period_hours, period_index

    assert [period_index(x)[0] for x in ["H1Q1", "H1Q4", "H2Q1", "H24Q4", "H5", "7"]] == [1, 4, 5, 96, 5, 7]
    bids = parse_curves(SAMPLE)
    assert len(bids) == 10
    assert set(bids["side"]) == {"supply", "demand"}
    assert bids["mw"].iloc[0] == 1500.0 and bids["price"].iloc[0] == 1500.0
    assert period_hours(compact(bids)).tolist() == [0.25, 0.25]


def test_effective_curves_keep_simple_offers_and_matched_extras():
    from pce68.data.omie_curves import compact, effective_curves, matched_price, parse_curves

    bids = compact(parse_curves(SAMPLE))
    first = effective_curves(bids)[0]
    # Offered simple supply (0 € and 50 €) plus the accepted part of the complex bid (300 MW at 10 €);
    # the 800 MW offered complex bid is dropped. Demand: 1500 MW simple plus 200 MW exported to France.
    assert sorted(zip(first.supply_price, first.supply_mw, strict=True)) == [(0, 1000), (10, 300), (50, 1000)]
    # Both demand bids are at 1500 €, so compacting merges them into one 1700 MW step.
    assert first.demand_price.tolist() == [1500.0] and first.demand_mw.tolist() == [1700.0]
    assert clear(first) == 50.0  # 1700 MW needed: 1300 MW up to 10 € is not enough
    assert matched_price(bids).tolist() == [50.0, 0.0]


def test_simulate_day_builds_the_training_panel():
    from pce68.data.omie_curves import compact, parse_curves
    from pce68.market.impact import simulate_day

    panel = simulate_day(compact(parse_curves(SAMPLE)), [0, 0.1], 1, 1.0, min_abs_price=5)
    assert len(panel) == 2 * 2  # periods x fleet sizes
    no_fleet = panel[panel["bess_gw"] == 0]
    assert (no_fleet["p_x"] == no_fleet["p0"]).all()  # calibration: X = 0 gives the historical price
    assert no_fleet["delta"].eq(0).all()
    # The second period clears at 0 €: no ratio there, only the additive impact.
    assert panel.loc[panel["period"] == 2, "ratio"].isna().all()
    assert panel.loc[panel["period"] == 1, "ratio"].notna().all()


# ── Older curve layouts (columns found by name)

LAYOUT_2024 = (
    "OMIE - Mercado de electricidad;Curvas agregadas;15/01/2024;\n"
    "Hora;Fecha;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);\n"
    "1;15/01/2024;V;1.000,5;0,00;O;\n"
    "1;15/01/2024;V;500,0;45,30;O;\n"
    "1;15/01/2024;C;2.000,0;180,00;O;\n"
    "1;15/01/2024;C;300,0;20,00;O;\n"
    "1;15/01/2024;V;900,0;45,30;C;\n"
)


def test_2024_layout_hourly_with_energy_column():
    from pce68.data.omie_curves import compact, effective_curves, matched_price, parse_curves

    bids = parse_curves(LAYOUT_2024)
    assert len(bids) == 5 and not bids["quarter_hourly"].any()
    day = compact(bids)
    # No bid-type column: the effective curve is every offered row.
    curve = effective_curves(day)[0]
    assert sorted(curve.supply_mw) == [500.0, 1000.5]
    assert matched_price(day).tolist() == [45.3]


def test_columns_are_found_by_name_not_position():
    from pce68.data.omie_curves import parse_curves

    shuffled = LAYOUT_2024.replace(
        "Hora;Fecha;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);",
        "Precio Compra/Venta;Hora;Fecha;Tipo Oferta;Energía Compra/Venta;Ofertada (O)/Casada (C);",
    ).replace("1;15/01/2024;V;1.000,5;0,00;O;", "0,00;1;15/01/2024;V;1.000,5;O;")
    assert parse_curves(shuffled)["price"].iloc[0] == 0.0


def test_tipologia_is_not_mistaken_for_tipo_oferta():
    from pce68.data.omie_curves import parse_curves

    bids = parse_curves(SAMPLE)  # 2025 layout: 'Tipo Oferta' and 'Tipología de Oferta'
    assert set(bids["side"]) == {"supply", "demand"}
    assert "S" in set(bids["kind"])


def test_a_page_that_is_not_a_curve_file_is_rejected():
    from pce68.data.omie_curves import parse_curves

    with pytest.raises(ValueError):
        parse_curves("<html><body>Not found</body></html>")
