"""Chart of the self-cannibalisation curve, for reports and slides.

    uv run python scripts/06_plot_fleet.py        # or `make plot`

Reads the table written by 05_fleet.py (data/processed/fleet.csv) and draws
two panels side by side, sharing the x axis (installed storage, GW):

    left   revenue per installed MW and year: the cannibalisation curve
    right  mean daily spread (highest minus lowest price of the day)

Two panels rather than one chart with two y axes: the measures have different
units and scales, and a second axis would make their crossing look meaningful.

Writes reports/cannibalisation_curve.png.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Colours by role (light surface).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
REFERENCE = "#8a8984"

PNIEC_GW = 17  # storage target of the Spanish 2030 NECP, marked on both panels


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _line(ax, x, y) -> None:
    ax.plot(x, y, color=SERIES, linewidth=2, solid_capstyle="round", zorder=3)
    ax.scatter(x, y, s=36, color=SERIES, edgecolors=SURFACE, linewidths=1.5, zorder=4)


def _label(ax, x, y, text, dx=0.4, dy=0.0, ha="left") -> None:
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), color=TEXT, fontsize=9, ha=ha, va="center")


def _pniec(ax) -> None:
    ax.axvline(PNIEC_GW, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(PNIEC_GW, 1.0, " PNIEC 2030", transform=ax.get_xaxis_transform(),
            color=TEXT_SECONDARY, fontsize=8.5, ha="left", va="bottom")  # fmt: skip


def plot(table: pd.DataFrame, days: str, output: Path) -> None:
    fleet = table[table["bess_gw"] > 0]
    gw, revenue = fleet["bess_gw"].to_numpy(), fleet["revenue_eur_per_mw_year"].to_numpy() / 1000
    spread = table["spread"].to_numpy()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, facecolor=SURFACE)
    for ax in (left, right):
        _style(ax)
        ax.set_xlabel("Battery storage installed in Spain (GW)", color=TEXT_SECONDARY, fontsize=9.5)
        ax.set_xlim(-0.5, max(table["bess_gw"].max(), PNIEC_GW) + 1)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    _line(left, gw, revenue)
    left.set_ylim(0, revenue.max() * 1.15)
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}k" if v else "0"))
    left.set_ylabel("EUR per MW per year", color=TEXT_SECONDARY, fontsize=9.5)
    left.set_title("Revenue of each MW of storage", color=TEXT, fontsize=11.5, loc="left", pad=14)
    first, at_target = fleet.iloc[0], fleet.loc[(fleet["bess_gw"] - PNIEC_GW).abs().idxmin()]
    _label(left, first.bess_gw, first.revenue_eur_per_mw_year / 1000,
           f"{first.revenue_eur_per_mw_year / 1000:,.0f}k at {first.bess_gw:g} GW", dy=6)  # fmt: skip
    share = f"({100 * at_target.vs_smallest_fleet:.0f}% of {first.bess_gw:g} GW)"
    _label(left, at_target.bess_gw, at_target.revenue_eur_per_mw_year / 1000,
           f"{at_target.revenue_eur_per_mw_year / 1000:,.0f}k at {at_target.bess_gw:g} GW\n{share}",
           dx=-0.6, dy=-9, ha="right")  # fmt: skip
    _pniec(left)

    _line(right, table["bess_gw"].to_numpy(), spread)
    right.set_ylim(0, spread.max() * 1.15)
    right.set_ylabel("EUR/MWh", color=TEXT_SECONDARY, fontsize=9.5)
    right.set_title("Mean daily price spread (max - min)", color=TEXT, fontsize=11.5, loc="left", pad=14)
    base, target_row = table.iloc[0], table.loc[(table["bess_gw"] - PNIEC_GW).abs().idxmin()]
    _label(right, base.bess_gw, base.spread, f"{base.spread:.0f} with no storage", dy=10)
    _label(right, target_row.bess_gw, target_row.spread,
           f"{target_row.spread:.0f} at {target_row.bess_gw:g} GW", dx=-0.6, dy=-12, ha="right")  # fmt: skip
    _pniec(right)

    fig.suptitle("Self-cannibalisation of battery storage in the Spanish day-ahead market",
                 x=0.012, ha="left", color=TEXT, fontsize=13.5, fontweight="bold")  # fmt: skip
    fig.text(0.012, 0.012,
             f"OMIE curves, {days}. 2 h batteries, 85% round-trip efficiency, perfect foresight, "
             "day-ahead energy only: an upper bound.",
             color=TEXT_SECONDARY, fontsize=8, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), w_pad=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    source = ROOT / config["data"]["processed_dir"] / "fleet.csv"
    if not source.exists():
        raise SystemExit(f"{source} not found; run scripts/05_fleet.py first.")
    table = pd.read_csv(source)

    impact_dir = ROOT / config["impact"]["output_dir"]
    stems = sorted(f.stem for f in impact_dir.glob("*.parquet"))
    days = (
        f"{pd.Timestamp(stems[0]):%d/%m/%Y} to {pd.Timestamp(stems[-1]):%d/%m/%Y} ({len(stems)} days)"
        if stems
        else "simulated days"
    )

    output = ROOT / "reports" / "cannibalisation_curve.png"
    plot(table, days, output)
    log.info("Wrote %s", output)


if __name__ == "__main__":
    main()
