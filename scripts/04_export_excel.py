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
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from pce68.dispatch.battery import Battery, simulate_price_taker
from pce68.features.metrics import daily_spread
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


def label_axes(ch: LineChart, x_title: str, y_title: str) -> None:
    """Give a chart axis titles and make sure Excel draws both axes.

    openpyxl 3.1 marks axes as deleted by default, and some Excel versions
    then hide them, labels and titles included.
    """
    ch.x_axis.title = x_title
    ch.y_axis.title = y_title
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.legend.position = "b"


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

    log.info("Computing daily spreads...")
    spread2 = daily_spread(prices, 2.0).rename(columns={"spread_2h": "spread"})
    spread2["year"] = pd.to_datetime(spread2["day_local"]).dt.year
    spread2["month"] = pd.to_datetime(spread2["day_local"]).dt.to_period("M").astype(str)

    log.info("Simulating battery dispatch...")
    revenue = simulate_price_taker(prices, bat)
    revenue["year"] = pd.to_datetime(revenue["day"]).dt.year

    # ── Annual
    annual = prices.groupby("year").apply(
        lambda g: pd.Series(
            {
                "Hours of data": g["hours"].sum(),
                "Mean price (€/MWh)": g["price_es"].mean(),
                "Median (€/MWh)": g["price_es"].median(),
                "Std. dev. (€/MWh)": g["price_es"].std(),
                "Minimum (€/MWh)": g["price_es"].min(),
                "Maximum (€/MWh)": g["price_es"].max(),
                "Hours at price ≤ 0": g.loc[g["price_es"] <= 0, "hours"].sum(),
            }
        ),
        include_groups=False,
    )
    annual["% hours at price ≤ 0"] = annual["Hours at price ≤ 0"] / annual["Hours of data"]
    annual["Mean 2 h spread (€/MWh)"] = spread2.groupby("year")["spread"].mean()
    annual["2 h battery revenue (€/MW/year)"] = revenue.groupby("year")["revenue_eur_per_mw"].sum()
    annual = annual.reset_index().rename(columns={"year": "Year"})
    annual["Year"] = annual["Year"].astype(str)

    # ── Monthly
    monthly = prices.groupby("month").apply(
        lambda g: pd.Series(
            {
                "Mean price (€/MWh)": g["price_es"].mean(),
                "Hours at price ≤ 0": g.loc[g["price_es"] <= 0, "hours"].sum(),
            }
        ),
        include_groups=False,
    )
    monthly["Mean 2 h spread (€/MWh)"] = spread2.groupby("month")["spread"].mean()
    monthly = monthly.reset_index().rename(columns={"month": "Month"})

    # ── Hourly profile
    profile = (
        prices.pivot_table(index="hour", columns="year", values="price_es", aggfunc="mean")
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
        "a person can read. Half a million quarter-hourly rows do not fit here in any useful way.",
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
            "Mean 2 h spread (€/MWh)": eur,
            "2 h battery revenue (€/MW/year)": "#,##0",
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
        {"Mean price (€/MWh)": eur, "Hours at price ≤ 0": "#,##0", "Mean 2 h spread (€/MWh)": eur},
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

    output = ROOT / "reports" / f"exploratory_analysis{'_demo' if args.demo else ''}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    log.info("\nWrote %s", output)
    log.info("\n%s", annual.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
