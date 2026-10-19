# ENGIE_68
Development project for ENGIE Spain
WE USE DATA FROM THE OMIE PLATFORM

Analysis of the Spanish day-ahead electricity market (OMIE) and of the arbitrage value of a battery.

## Usage

```bash
uv sync
make data         # download OMIE -> data/processed/omie_prices.parquet
make excel        # reports/exploratory_analysis.xlsx
make excel-demo   # the same with synthetic prices, no download needed
make test
```

Without `make` (e.g. on Windows):

```bash
python scripts/01_download_omie.py
python scripts/04_export_excel.py          # add --demo for synthetic prices
python -m pytest -q
```

## Layout

| Path | What it does |
|---|---|
| `config/base.yaml` | Parameters: download dates and battery (duration, efficiency, cycles). |
| `src/pce68/data/omie.py` | Download and parsing of `marginalpdbc` files (hourly and quarter-hourly, daylight-saving days, yearly zips). |
| `src/pce68/features/regimes.py` | Labels each period with its regime: pre_crisis, gas_crisis, iberian_exception, normalization. |
| `src/pce68/features/metrics.py` | Daily spread: mean of the N most expensive hours minus the N cheapest. |
| `src/pce68/dispatch/battery.py` | Arbitrage of 1 MW of battery, price taker, one cycle per day. |
| `scripts/` | Numbered entry points of the pipeline. |

The assumptions behind each calculation are documented at the top of its module.
