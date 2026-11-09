"""Export the exploratory analysis to Excel.

    uv run python scripts/04_export_excel.py            # with the data from `make data`
    uv run python scripts/04_export_excel.py --demo     # synthetic prices, for testing

The file it produces is an OUTPUT, not a store: it is regenerated in full
every time and never edited by hand. The raw data lives in data/processed as
parquet, which is what the analysis and the models read.

Sheets:
    Read me         what this is and when it was generated
    By regime       why the model cannot be trained on the whole series
    Annual          price, dispersion, zero-price hours and spread per year
    Monthly         the same series by month, with a chart
    Hourly profile  mean price per hour and year: the duck curve
    Cannibalisation revenue per MW and spread against GW of storage, from
                    05_fleet.py (only if data/processed/fleet.csv exists:
                    run 03_simulate_impact.py and 05_fleet.py before this)
    Impact model    stage 2: how well the statistical model reproduces the
                    simulator on months it did not see, and the revenue curve
                    from its prices (only if 08_fit_impact_model.py has run)
    Projection      stage 3: the price forecast adjusted for each storage
                    scenario (only if 09_project.py has run)
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from pce68.dispatch.battery import Battery, simulate_price_taker
from pce68.features.metrics import daily_spread, weighted_mean, weighted_median, weighted_std
from pce68.features.regimes import annotate, summary

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BLUE = "1F3864"
F_TITLE = Font(name="Arial", size=14, bold=True, color=BLUE)
F_HDR = Font(name="Arial", size=10, bold=True)
F_TXT = Font(name="Arial", size=10)
F_NOTE = Font(name="Arial", size=9, italic=True, color="666666")
FILL_HDR = PatternFill("solid", fgColor="DCE6F1")


def demo_prices(years: range) -> pd.DataFrame:
    """Synthetic series with a solar shape and a different level per regime."""
    rng = np.random.default_rng(7)
    frames = []
    for year in years:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq="15min", tz="Europe/Madrid")
        h = idx.hour + idx.minute / 60
        level = {2021: 90, 2022: 180, 2023: 110, 2024: 65, 2025: 60, 2026: 55}.get(year, 60)
        trough = 45 * np.clip(np.sin(np.pi * (h - 7) / 12), 0, None) * (1 + (year - 2021) * 0.12)
        peak = 20 * np.exp(-((h - 21) ** 2) / 3)
        p = level - trough + peak + rng.normal(0, 8, len(idx))
        frames.append(
            pd.DataFrame({"timestamp_utc": idx.tz_convert("UTC"), "price_es": p, "resolution_min": 15})
        )
    return pd.concat(frames, ignore_index=True)


def cell_value(v):
    """Turn a pandas/numpy value into something openpyxl writes correctly.

    NaN must come out as an empty cell: openpyxl writes it literally and Excel
    then opens the file as damaged. In object columns NaN arrives as a Python
    float, not a numpy one, so any scalar is checked.
    """
    if v is None or (np.isscalar(v) and pd.isna(v)) or v is pd.NaT:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.tz_localize(None) if v.tzinfo else v
    return v


def table(ws, row0, df: pd.DataFrame, formats: dict[str, str] | None = None) -> int:
    """Write a DataFrame as a table with a header. Returns the last row."""
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=row0, column=j, value=str(col))
        c.font, c.fill = F_HDR, FILL_HDR
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = max(12, min(26, len(str(col)) + 4))

    for i, (_, r) in enumerate(df.iterrows(), start=row0 + 1):
        for j, col in enumerate(df.columns, start=1):
            c = ws.cell(row=i, column=j, value=cell_value(r[col]))
            c.font = F_TXT
            if formats and col in formats:
                c.number_format = formats[col]
    return row0 + len(df)


def label_axes(ch: LineChart | ScatterChart, x_title: str, y_title: str) -> None:
    """Give a chart axis titles and make sure Excel draws both axes.

    openpyxl 3.1 marks axes as deleted by default, and some Excel versions
    then hide them, labels and titles included.
    """
    ch.x_axis.title = x_title
    ch.y_axis.title = y_title
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.legend.position = "b"


def add_cannibalisation_sheet(wb: Workbook, fleet: pd.DataFrame, eur: str, pct: str) -> None:
    """Sheet with the table of 05_fleet.py and its two curves, as native Excel charts.

    Scatter charts with lines, not line charts: fleet sizes are unevenly
    spaced (0, 1, 2, 4, 6 ... GW) and a line chart would draw them evenly.
    Two charts rather than one with two y axes, as the units differ.
    """
    ws = wb.create_sheet("Cannibalisation")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Self-cannibalisation of battery storage"
    ws["A1"].font = F_TITLE
    ws["A2"] = (
        "Revenue of each MW of storage as more storage is installed in Spain. OMIE curves, "
        "perfect foresight, day-ahead energy only: an upper bound. PNIEC 2030 target: about 17 GW."
    )
    ws["A2"].font = F_NOTE
    view = pd.DataFrame(
        {
            "Storage installed (GW)": fleet["bess_gw"],
            "Revenue (€/MW/year)": fleet["revenue_eur_per_mw_year"],
            "vs smallest fleet": fleet["vs_smallest_fleet"],
            "Mean daily spread (€/MWh)": fleet["spread"],
            "Mean price (€/MWh)": fleet["mean_price"],
        }
    )
    first, last = (
        5,
        table(
            ws,
            4,
            view,
            {
                "Storage installed (GW)": "0.##",
                "Revenue (€/MW/year)": "#,##0",
                "vs smallest fleet": pct,
                "Mean daily spread (€/MWh)": eur,
                "Mean price (€/MWh)": eur,
            },
        ),
    )

    for col, title, y_title, anchor in (
        (2, "Revenue of each MW of storage (€/MW/year)", "€ per MW per year", "G4"),
        (4, "Mean daily price spread, max - min (€/MWh)", "€/MWh", "G24"),
    ):
        # Revenue is empty at 0 GW (no fleet to divide by): start the series at its first value.
        start = first + int(view.iloc[:, col - 1].isna().idxmin())
        ch = ScatterChart()
        ch.title = title
        ch.style = 13
        ch.height, ch.width = 9, 18
        x = Reference(ws, min_col=1, min_row=start, max_row=last)
        series = Series(Reference(ws, min_col=col, min_row=start, max_row=last), x, title=title)
        series.marker.symbol = "circle"
        series.marker.size = 7
        series.smooth = False
        ch.series.append(series)
        ch.y_axis.scaling.min = 0
        ch.x_axis.scaling.min = 0
        label_axes(ch, "Battery storage installed in Spain (GW)", y_title)
        ch.legend = None
        ws.add_chart(ch, anchor)


def scatter(
    ws,
    title: str,
    y_title: str,
    x_ref: tuple[int, int, int],
    columns: list[int],
    anchor: str,
    x_title: str = "Battery storage installed in Spain (GW)",
    x_range: tuple[float, float] | None = None,
):
    """Scatter chart with lines and markers: x in column x_ref[0], rows x_ref[1]..x_ref[2],
    one series per column, titled from the header row just above."""
    col_x, first, last = x_ref
    ch = ScatterChart()
    ch.title = title
    ch.style = 13
    ch.height, ch.width = 9, 18
    x = Reference(ws, min_col=col_x, min_row=first, max_row=last)
    for col in columns:
        series = Series(Reference(ws, min_col=col, min_row=first - 1, max_row=last), x, title_from_data=True)
        series.marker.symbol = "circle"
        series.marker.size = 6
        series.smooth = False
        ch.series.append(series)
    ch.y_axis.scaling.min = 0
    ch.x_axis.scaling.min = 0
    if x_range:  # e.g. years: the axis would otherwise start at 0
        ch.x_axis.scaling.min, ch.x_axis.scaling.max = x_range
        ch.x_axis.majorUnit = 1
    label_axes(ch, x_title, y_title)
    ws.add_chart(ch, anchor)


def add_projection_sheet(
    wb: Workbook, summary: pd.DataFrame, curve: pd.DataFrame, source: str, eur: str
) -> None:
    """Sheet with the stage-3 results of 09_project.py: the BESS-adjusted forecast by scenario."""
    ws = wb.create_sheet("Projection")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Stage 3 · Price forecast adjusted for battery storage"
    ws["A1"].font = F_TITLE
    ws["A2"] = source
    ws["A2"].font = Font(name="Arial", size=10, bold=True, color="C00000" if "TEST" in source else "1F3864")
    notes = [
        "For each storage scenario the impact model gives y = f(t, X, z) and the adjusted forecast "
        "p' = y · p (roadmap, step 3). Scenario paths are in config/base.yaml (projection.scenarios).",
        "Revenue: 1 MW of 2 h battery trading on the prices (price taker, perfect foresight, one cycle a "
        "day, day-ahead only): what one more MW would earn with X GW installed. An upper bound.",
    ]
    for i, text in enumerate(notes, start=3):
        ws.cell(row=i, column=1, value=text).font = F_NOTE

    # 1. By scenario and year
    row = 6
    ws.cell(row=row, column=1, value="1. By scenario and year").font = F_TITLE
    view = summary.rename(
        columns={
            "scenario": "Scenario",
            "year": "Year",
            "GW (mean)": "Storage (GW, mean)",
            "mean price base": "Mean price, base (€/MWh)",
            "mean price adjusted": "Mean price, adjusted (€/MWh)",
            "daily spread base": "Daily spread, base (€/MWh)",
            "daily spread adjusted": "Daily spread, adjusted (€/MWh)",
            "revenue base (EUR/MW/yr)": "Revenue, base (€/MW/yr)",
            "revenue adjusted (EUR/MW/yr)": "Revenue, adjusted (€/MW/yr)",
        }
    )
    formats = {c: (eur if "€/MWh" in c else "#,##0") for c in view.columns[2:]}
    formats |= {"Year": "0", "Storage (GW, mean)": "0.0"}
    last = table(ws, row + 1, view, formats)

    # 2. Revenue by year, one column per scenario
    row = last + 3
    ws.cell(row=row, column=1, value="2. Revenue of 1 MW of storage by scenario (€/MW/yr)").font = F_TITLE
    pivot = summary.pivot(index="year", columns="scenario", values="revenue adjusted (EUR/MW/yr)")
    pivot = pivot[list(dict.fromkeys(summary["scenario"]))].reset_index().rename(columns={"year": "Year"})
    first = row + 2
    last = table(ws, row + 1, pivot, {"Year": "0"} | dict.fromkeys(pivot.columns[1:], "#,##0"))
    columns = list(range(2, pivot.shape[1] + 1))
    scatter(ws, "Revenue of 1 MW of storage by scenario", "€ per MW per year", (1, first, last), columns,
            "L6", x_title="Year", x_range=(pivot["Year"].min(), pivot["Year"].max()))  # fmt: skip

    # 3. Cannibalisation curve on the forecast: fixed sizes, one column per year
    row = last + 3
    ws.cell(
        row=row, column=1, value="3. Revenue of 1 MW with X GW installed, by year (€/MW/yr)"
    ).font = F_TITLE
    pivot = curve.pivot(index="GW", columns="year", values="revenue (EUR/MW/yr)").reset_index()
    pivot.columns = ["Storage installed (GW)", *[str(c) for c in pivot.columns[1:]]]
    first = row + 2
    last = table(
        ws, row + 1, pivot, {"Storage installed (GW)": "0.##"} | dict.fromkeys(pivot.columns[1:], "#,##0")
    )
    columns = list(range(2, pivot.shape[1] + 1))
    scatter(
        ws, "Cannibalisation curve on the forecast", "€ per MW per year", (1, first, last), columns, "L26"
    )
    ws.column_dimensions["A"].width = 30
    for j in range(2, 11):
        ws.column_dimensions[get_column_letter(j)].width = 16


def add_impact_model_sheet(
    wb: Workbook, scores: pd.DataFrame, by_gw: pd.DataFrame, revenue: pd.DataFrame, eur: str, pct: str
) -> None:
    """Sheet with the stage-2 validation written by 08_fit_impact_model.py.

    The best model is the one with the smallest mean revenue error over the
    fleet sizes, the same rule the script uses to choose the model it saves.
    """
    errors = revenue.drop(columns=["GW", "simulated"]).div(revenue["simulated"], axis=0) - 1
    ranking = errors.drop(columns="no impact").abs().mean().sort_values()
    best = ranking.index[0]
    base = best.removesuffix(" + calibrated")

    ws = wb.create_sheet("Impact model")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Stage 2 · Statistical model of the price impact of storage"
    ws["A1"].font = F_TITLE
    notes = [
        "The model learns from the simulated days how much X GW of storage move the price of each quarter "
        "hour, so that it can be applied to a price forecast, where there are no OMIE curves.",
        "Every figure is on months the model did not see: each month is predicted by a model fitted on "
        "the others (leave one month out).",
        f"Best model: {best}, chosen by the smallest revenue error over fleet sizes.",
    ]
    for i, text in enumerate(notes, start=2):
        ws.cell(row=i, column=1, value=text).font = F_NOTE

    # 1. Price error per model
    row = 6
    ws.cell(row=row, column=1, value="1. Error of the predicted price with storage").font = F_TITLE
    view = scores.rename(columns={"model": "Model"})
    row = table(ws, row + 1, view, {c: (eur if "MAE" in c else "0.000") for c in view.columns[1:]}) + 1
    note = (
        "MAE in €/MWh. Cheapest / dearest 10%: the periods of each day where storage charges / "
        "discharges. R²: share of the impact explained (1 = all)."
    )
    ws.cell(row=row, column=1, value=note).font = F_NOTE

    # 2. Revenue curve: simulated against the model's prices
    row += 3
    ws.cell(row=row, column=1, value="2. Revenue of each MW of storage (€/MW/year)").font = F_TITLE
    curves = {"Simulated (OMIE curves)": revenue["simulated"], f"Model: {best}": revenue[best]}
    if base != best:
        curves[f"Before calibration: {base}"] = revenue[base]
    curves["Ignoring cannibalisation"] = revenue["no impact"]
    gaps = {f"Error: {best}": errors[best]}
    if base != best:
        gaps["Error before calibration"] = errors[base]
    gaps["Error ignoring cannibalisation"] = errors["no impact"]
    view = pd.DataFrame({"Storage installed (GW)": revenue["GW"]} | curves | gaps)
    first = row + 2
    formats = {"Storage installed (GW)": "0.##"} | dict.fromkeys(curves, "#,##0") | dict.fromkeys(gaps, pct)
    last = table(ws, row + 1, view, formats)
    note = (
        "The same simulated dispatch priced three ways. Spain-Portugal splits are left out, so the "
        "simulated column can differ slightly from the Cannibalisation sheet."
    )
    ws.cell(row=last + 1, column=1, value=note).font = F_NOTE
    title = "Revenue of each MW of storage (€/MW/year)"
    scatter(ws, title, "€ per MW per year", (1, first, last), list(range(2, len(curves) + 2)), "K6")

    # 3. Price error by fleet size
    row = last + 4
    ws.cell(row=row, column=1, value="3. Price error by fleet size, MAE (€/MWh)").font = F_TITLE
    candidates = dict.fromkeys([best, base, "ridge", "rank lookup", "no impact"])
    shown = [m for m in candidates if m in by_gw.columns]
    view = by_gw[["GW", *shown]].rename(columns={"GW": "Storage installed (GW)"})
    first = row + 2
    last = table(ws, row + 1, view, {c: eur for c in shown} | {"Storage installed (GW)": "0.##"})
    columns = list(range(2, len(shown) + 2))
    scatter(ws, "Error of the predicted price by fleet size", "MAE (€/MWh)", (1, first, last), columns, "K26")
    ws.column_dimensions["A"].width = 40
    for j in range(2, 10):
        ws.column_dimensions[get_column_letter(j)].width = 17


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    bat = Battery(
        duration_hours=config["battery"]["duration_hours"],
        round_trip_efficiency=config["battery"]["round_trip_efficiency"],
        max_cycles_per_year=config["battery"]["max_cycles_per_year"],
    )

    if args.demo:
        log.info("DEMO MODE: synthetic prices, not real data.")
        prices = demo_prices(range(2021, 2027))
        source = "DEMO · synthetic prices. NOT market data."
    else:
        path = ROOT / config["data"]["processed_dir"] / "omie_prices.parquet"
        if not path.exists():
            raise SystemExit(f"{path} does not exist. Run `make data` first, or use --demo.")
        prices = pd.read_parquet(path)
        source = "OMIE day-ahead market prices, downloaded with `make data`."

    prices = annotate(prices)
    local = prices["timestamp_utc"].dt.tz_convert("Europe/Madrid")
    prices["year"] = local.dt.year
    prices["month"] = local.dt.strftime("%Y-%m")
    prices["hour"] = local.dt.hour
    prices["hours"] = prices["resolution_min"] / 60.0

    # Spread and battery use the same duration, so the two columns compare like with like.
    dur = bat.duration_hours
    spread_col = f"Mean {dur:g} h spread (€/MWh)"
    revenue_col = f"{dur:g} h battery revenue (€/MW/year)"

    log.info("Computing daily spreads...")
    spread2 = daily_spread(prices, dur).rename(columns={f"spread_{dur:g}h": "spread"})
    spread2["year"] = pd.to_datetime(spread2["day_local"]).dt.year
    spread2["month"] = pd.to_datetime(spread2["day_local"]).dt.to_period("M").astype(str)

    log.info("Simulating battery dispatch...")
    revenue = simulate_price_taker(prices, bat)
    revenue["year"] = pd.to_datetime(revenue["day"]).dt.year

    # ── Annual
    # Every statistic is weighted by period length: since 01/10/2025 there are four
    # quarter-hour rows per hour, and an unweighted mean would count them four times.
    annual = prices.groupby("year").apply(
        lambda g: pd.Series(
            {
                "Hours of data": g["hours"].sum(),
                "Mean price (€/MWh)": weighted_mean(g["price_es"], g["hours"]),
                "Median (€/MWh)": weighted_median(g["price_es"], g["hours"]),
                "Std. dev. (€/MWh)": weighted_std(g["price_es"], g["hours"]),
                "Minimum (€/MWh)": g["price_es"].min(),
                "Maximum (€/MWh)": g["price_es"].max(),
                "Hours at price ≤ 0": g.loc[g["price_es"] <= 0, "hours"].sum(),
            }
        ),
        include_groups=False,
    )
    annual["% hours at price ≤ 0"] = annual["Hours at price ≤ 0"] / annual["Hours of data"]
    annual[spread_col] = spread2.groupby("year")["spread"].mean()
    annual[revenue_col] = revenue.groupby("year")["revenue_eur_per_mw"].sum()
    annual = annual.reset_index().rename(columns={"year": "Year"})
    annual["Year"] = annual["Year"].astype(str)

    # ── Monthly
    monthly = prices.groupby("month").apply(
        lambda g: pd.Series(
            {
                "Mean price (€/MWh)": weighted_mean(g["price_es"], g["hours"]),
                "Hours at price ≤ 0": g.loc[g["price_es"] <= 0, "hours"].sum(),
            }
        ),
        include_groups=False,
    )
    monthly[spread_col] = spread2.groupby("month")["spread"].mean()
    monthly = monthly.reset_index().rename(columns={"month": "Month"})

    # ── Hourly profile
    profile = (
        prices.groupby(["hour", "year"])
        .apply(lambda g: weighted_mean(g["price_es"], g["hours"]), include_groups=False)
        .unstack("year")
        .reset_index()
        .rename(columns={"hour": "Hour"})
    )
    profile.columns = [str(c) for c in profile.columns]

    # ── Regimes
    reg = summary(prices)
    reg = reg.rename(
        columns={
            "regime": "Regime",
            "periods": "Periods",
            "start": "From",
            "end": "To",
            "mean_price": "Mean price (€/MWh)",
            "max_price": "Maximum price (€/MWh)",
        }
    )

    # ── Writing
    wb = Workbook()
    eur = "#,##0.00"
    pct = "0.0%"

    ws = wb.active
    ws.title = "Read me"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 100
    ws["A1"] = "Project 68 · Exploratory analysis of the day-ahead market"
    ws["A1"].font = F_TITLE
    texts = [
        f"Generated on {date.today():%d/%m/%Y} by scripts/04_export_excel.py",
        source,
        "",
        "This file is an OUTPUT: it is regenerated in full every time the script runs.",
        "Do not edit it by hand, because the next `make` overwrites any changes.",
        "",
        "The raw data lives in data/processed/omie_prices.parquet. The analysis and the",
        "machine-learning models read from there; Excel only gets the aggregates, which are what",
        "a person can read.",
        "",
        "Every price statistic is weighted by period length, so the quarter-hourly market",
        "(from 01/10/2025) and the hourly one before it count the same per hour.",
        "",
        "The figures are values computed in Python, not formulas: traceability lives in the code",
        "and in version control, not in the cells.",
    ]
    for i, t in enumerate(texts):
        ws.cell(row=3 + i, column=1, value=t).font = F_TXT if t else F_NOTE

    ws = wb.create_sheet("By regime")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "The five years of data are not a single market"
    ws["A1"].font = F_TITLE
    ws["A2"] = (
        "Training on the whole series mixes regimes that no longer exist. "
        'Only "normalization" represents the market we project to 2031.'
    )
    ws["A2"].font = F_NOTE
    table(ws, 4, reg, {"Mean price (€/MWh)": eur, "Maximum price (€/MWh)": eur})

    ws = wb.create_sheet("Annual")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Annual summary"
    ws["A1"].font = F_TITLE
    table(
        ws,
        3,
        annual,
        {
            "Mean price (€/MWh)": eur,
            "Median (€/MWh)": eur,
            "Std. dev. (€/MWh)": eur,
            "Minimum (€/MWh)": eur,
            "Maximum (€/MWh)": eur,
            "Hours of data": "#,##0",
            "Hours at price ≤ 0": "#,##0",
            "% hours at price ≤ 0": pct,
            spread_col: eur,
            revenue_col: "#,##0",
        },
    )

    ws = wb.create_sheet("Monthly")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Monthly series"
    ws["A1"].font = F_TITLE
    last = table(
        ws,
        3,
        monthly,
        {"Mean price (€/MWh)": eur, "Hours at price ≤ 0": "#,##0", spread_col: eur},
    )
    ch = LineChart()
    ch.title = "Mean price and capturable spread (€/MWh)"
    ch.height, ch.width = 9, 24
    ch.add_data(Reference(ws, min_col=2, max_col=2, min_row=3, max_row=last), titles_from_data=True)
    ch.add_data(Reference(ws, min_col=4, max_col=4, min_row=3, max_row=last), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=4, max_row=last))
    label_axes(ch, "Month", "Price / spread (€/MWh)")
    ws.add_chart(ch, "F3")

    ws = wb.create_sheet("Hourly profile")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Mean price by hour of day"
    ws["A1"].font = F_TITLE
    ws["A2"] = "The midday trough that deepens year after year is solar cannibalisation."
    ws["A2"].font = F_NOTE
    last = table(ws, 4, profile, {c: eur for c in profile.columns if c != "Hour"})
    ch = LineChart()
    ch.title = "Hourly price profile by year (€/MWh)"
    ch.height, ch.width = 10, 24
    ch.add_data(
        Reference(ws, min_col=2, max_col=len(profile.columns), min_row=4, max_row=last),
        titles_from_data=True,
    )
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=last))
    label_axes(ch, "Hour of day (local time, 0-23)", "Mean price (€/MWh)")
    ws.add_chart(ch, get_column_letter(len(profile.columns) + 2) + "4")

    fleet_file = ROOT / config["data"]["processed_dir"] / "fleet.csv"
    if args.demo:
        pass  # the fleet table comes from real curves, not from the synthetic prices
    elif fleet_file.exists():
        add_cannibalisation_sheet(wb, pd.read_csv(fleet_file), eur, pct)
    else:
        log.info(
            "No %s: run 03_simulate_impact.py and 05_fleet.py to add the Cannibalisation sheet.",
            fleet_file.name,
        )

    reports = ROOT / "reports"
    model_files = [reports / f"impact_model_{n}.csv" for n in ("scores", "by_gw", "revenue")]
    if args.demo:
        pass
    elif all(f.exists() for f in model_files):
        add_impact_model_sheet(wb, *(pd.read_csv(f) for f in model_files), eur, pct)
    else:
        log.info("No impact model results: run 08_fit_impact_model.py to add the Impact model sheet.")

    projection = [reports / f"projection_{n}" for n in ("summary.csv", "curve.csv", "source.txt")]
    if args.demo:
        pass
    elif all(f.exists() for f in projection):
        proj_summary, proj_curve = (pd.read_csv(f) for f in projection[:2])
        add_projection_sheet(wb, proj_summary, proj_curve, projection[2].read_text().strip(), eur)
    else:
        log.info("No projection: run 09_project.py to add the Projection sheet.")

    output = ROOT / "reports" / f"exploratory_analysis{'_demo' if args.demo else ''}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    log.info("\nWrote %s", output)
    log.info("\n%s", annual.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
