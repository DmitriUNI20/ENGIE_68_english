"""Find out why re-clearing the OMIE curves does not reproduce the published price.

    .venv\\Scripts\\python.exe scripts/99_diagnose_curves.py

Throwaway diagnostic, not part of the pipeline. It is deliberately standalone:
it parses the file itself rather than going through pce68.data.omie_curves, so
a bug in that reader cannot hide the evidence.

It prints what the file actually contains (which countries, which states, which
bid types) and then re-clears one day under every combination of filters, so
the combination that reproduces marginalpdbc identifies itself.
"""

from __future__ import annotations

import itertools
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data" / "raw" / "omie_curves"
PRICES = ROOT / "data" / "raw" / "omie"
PERIODS_SHOWN = [1, 8, 13, 20]


def fold(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def number(text: str) -> float:
    c = text.strip().replace(".", "").replace(",", ".")
    return float("nan") if c in ("", "-") else float(c)


def read(path: Path) -> tuple[list[str], list[list[str]]]:
    """Header fields and data rows, split on semicolons."""
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    i = next(j for j, ln in enumerate(lines[:20]) if "precio" in fold(ln))
    header = [h.strip() for h in lines[i].split(";")]
    rows = [r.split(";") for r in lines[i + 1 :] if r.count(";") >= len(header) - 2 and r[:1].isdigit()]
    return header, rows


def column(header: list[str], *fragments: str) -> int:
    for j, h in enumerate(header):
        if any(f in fold(h) for f in fragments):
            return j
    raise SystemExit(f"no column matching {fragments} in {header}")


def published(day: str) -> dict[int, float]:
    """period -> Spanish price, from the marginalpdbc file of that day."""
    path = PRICES / f"marginalpdbc_{day}.1"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/01_download_omie.py first.")
    out = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line[:1].isdigit():
            continue
        f = line.split(";")
        out[int(f[3])] = float(f[5])  # year;month;day;period;PT;ES
    return out


def clear(bids: list[tuple[str, float, float]]) -> float:
    """Clearing price of (kind, quantity, price) rows: lowest price where supply covers demand."""
    sell = sorted([(p, q) for k, q, p in bids if k == "V"])
    buy = sorted([(p, q) for k, q, p in bids if k == "C"], reverse=True)
    if not sell or not buy:
        return float("nan")

    sell_p = np.array([p for p, _ in sell])
    sell_v = np.cumsum([q for _, q in sell])
    buy_p = np.array([p for p, _ in buy])
    buy_v = np.cumsum([q for _, q in buy])

    for p in np.unique(np.concatenate([sell_p, buy_p])):
        s = sell_v[np.searchsorted(sell_p, p, side="right") - 1] if p >= sell_p[0] else 0.0
        i = np.searchsorted(-buy_p, -p, side="right") - 1
        b = buy_v[i] if i >= 0 else 0.0
        if s >= b:
            return float(p)
    return float("nan")


def main() -> None:
    files = sorted(CURVES.glob("curva_pbc_*.1"))
    if not files:
        raise SystemExit(f"no curve files in {CURVES}")
    path = files[len(files) // 2]
    day = path.stem.split("_")[-1]
    header, rows = read(path)

    print(f"\nFile:   {path.name}")
    print(f"Header: {';'.join(header)}")
    print(f"Rows:   {len(rows)}")

    c_period = column(header, "periodo", "hora")
    c_kind = column(header, "tipo oferta")
    c_qty = column(header, "energ", "potencia")
    c_price = column(header, "precio")
    try:
        c_country = column(header, "pais")
    except SystemExit:
        c_country = None
    try:
        c_state = column(header, "ofertada", "casada")
    except SystemExit:
        c_state = None
    c_class = None
    for j, h in enumerate(header):
        if "tipolog" in fold(h):
            c_class = j

    def values(col: int | None) -> dict[str, int]:
        if col is None:
            return {}
        out: dict[str, int] = {}
        for r in rows:
            out[r[col].strip()] = out.get(r[col].strip(), 0) + 1
        return out

    print("\nWhat the columns actually contain")
    print(f"  country  {values(c_country)}")
    print(f"  state    {values(c_state)}")
    print(f"  kind     {values(c_kind)}")
    if c_class is not None:
        print(f"  class    {values(c_class)}")
    print(f"  periods  {len({int(r[c_period]) for r in rows})}")

    real = published(day)
    countries = sorted(values(c_country)) or [None]
    states = sorted(values(c_state)) or [None]

    shown_prices = "  ".join(f"p{p}={real.get(p, float('nan')):.2f}" for p in PERIODS_SHOWN)
    shown_heads = "  ".join(f"{'p' + str(p):>9}" for p in PERIODS_SHOWN)
    print(f"\nRe-clearing against marginalpdbc of {day}")
    print(f"  published prices: {shown_prices}")
    print(f"\n  {'country':>10}  {'state':>6}  {shown_heads}   mean |err|")

    best = None
    for country, state in itertools.product(countries + ["(all)"], states + ["(all)"]):
        errors, shown = [], []
        for period in sorted({int(r[c_period]) for r in rows}):
            picked = [
                (r[c_kind].strip(), number(r[c_qty]), number(r[c_price]))
                for r in rows
                if int(r[c_period]) == period
                and (country == "(all)" or c_country is None or r[c_country].strip() == country)
                and (state == "(all)" or c_state is None or r[c_state].strip() == state)
            ]
            price = clear(picked) if picked else float("nan")
            if period in PERIODS_SHOWN:
                shown.append(price)
            if period in real and np.isfinite(price):
                errors.append(abs(price - real[period]))
        mean_err = float(np.mean(errors)) if errors else float("nan")
        cells = "  ".join(f"{v:>9.2f}" for v in shown)
        print(f"  {str(country):>10}  {str(state):>6}  {cells}   {mean_err:>10.3f}")
        if np.isfinite(mean_err) and (best is None or mean_err < best[0]):
            best = (mean_err, country, state)

    if best:
        print(f"\n  Best: country={best[1]} state={best[2]} with a mean error of {best[0]:.3f} EUR/MWh")
        print("  Anything under ~0.5 is the right combination; the reference reports 0.05.")
        print("  CAUTION: state=C (matched) reproduces the price by construction and carries no")
        print("  information beyond the clearing point, which is the region the method explores.")

    # ── Which side is short, at the price OMIE actually published
    print("\nBalance at the published price (volumes in MWh)")
    print(
        f"  {'period':>6}  {'state':>6}  {'n buy':>7}  {'n sell':>7}  {'buy tot':>10}  {'sell tot':>10}"
        f"  {'buy@pub':>10}  {'sell@pub':>10}  {'sell-buy':>10}"
    )
    for period in PERIODS_SHOWN:
        if period not in real:
            continue
        p_pub = real[period]
        for state in states + ["(all)"]:
            picked = [
                (r[c_kind].strip(), number(r[c_qty]), number(r[c_price]))
                for r in rows
                if int(r[c_period]) == period
                and (state == "(all)" or c_state is None or r[c_state].strip() == state)
            ]
            buy = [(p, q) for k, q, p in picked if k == "C"]
            sell = [(p, q) for k, q, p in picked if k == "V"]
            buy_at = sum(q for p, q in buy if p >= p_pub)
            sell_at = sum(q for p, q in sell if p <= p_pub)
            print(
                f"  {period:>6}  {str(state):>6}  {len(buy):>7}  {len(sell):>7}"
                f"  {sum(q for _, q in buy):>10,.0f}  {sum(q for _, q in sell):>10,.0f}"
                f"  {buy_at:>10,.0f}  {sell_at:>10,.0f}  {sell_at - buy_at:>10,.0f}"
            )
    print("\n  On the true offered curve the last two columns should match at the published price,")
    print("  so 'sell-buy' near zero identifies the curve the method needs. A large positive")
    print("  number means surplus supply, which is what drags the clearing price down.")

    # ── Are the matched rows a copy of the offered ones, or different bids?
    period = PERIODS_SHOWN[0]
    def triples(state: str) -> set[tuple[str, float, float]]:
        return {
            (r[c_kind].strip(), number(r[c_qty]), number(r[c_price]))
            for r in rows
            if int(r[c_period]) == period and c_state is not None and r[c_state].strip() == state
        }

    if c_state is not None and {"O", "C"} <= set(values(c_state)):
        offered, matched = triples("O"), triples("C")
        print(f"\nOverlap of the two sets in period {period}")
        print(f"  distinct offered rows  {len(offered)}")
        print(f"  distinct matched rows  {len(matched)}")
        print(f"  identical in both      {len(offered & matched)}")
        print("  Many identical rows mean the file lists each bid twice and the two sets must not")
        print("  be added together; few mean the matched rows are partial fills of the offered ones.")


if __name__ == "__main__":
    sys.exit(main())
