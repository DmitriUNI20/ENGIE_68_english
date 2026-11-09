# ENGIE_68
Development project for ENGIE Spain
WE USE DATA FROM THE OMIE PLATFORM

Analysis of the Spanish day-ahead electricity market (OMIE), the arbitrage value of a battery,
and how that value shrinks as more storage is installed (self-cannibalisation).

## Pipeline

Everything, in order, ending with the Excel report (each step skips what is already done):

```bash
python scripts/run_all.py              # or `make all`; add --sample 3 for a faster first run
```

Step by step:

| Step | Command (Windows: `python scripts/...`) | Output |
|---|---|---|
| 1. Prices | `make data` · `01_download_omie.py` | `data/processed/omie_prices.parquet` |
| 2. Curves | `make curves` · `02_download_curves.py` (`--sample 3`, `--start`, `--check`) | `data/processed/curves/curves_YYYYMMDD.parquet` |
| 3. Impact | `make impact` · `03_simulate_impact.py` | `data/processed/impact/YYYYMMDD.parquet` |
| 4. Excel | `make excel` · `04_export_excel.py` (`--demo` for synthetic prices) | `reports/exploratory_analysis.xlsx`, with a Cannibalisation sheet once step 5 has run |
| 5. Fleet | `make fleet` · `05_fleet.py` | cannibalisation table, `data/processed/fleet.csv` |
| 6. Chart | `make plot` · `06_plot_fleet.py` | `reports/cannibalisation_curve.png` |

`make test` runs the tests. Parameters live in `config/base.yaml`.

## Layout

| Path | What it does |
|---|---|
| `config/base.yaml` | Parameters: download dates, battery (duration, efficiency, cycles), curves and fleet sizes. |
| `src/pce68/data/omie.py` | Download and parsing of `marginalpdbc` price files (hourly and quarter-hourly, daylight-saving days, yearly zips). |
| `src/pce68/data/omie_curves.py` | Parsing of OMIE's aggregated curves (`curva_pbc`, every known layout), the effective curve and the compact format. |
| `src/pce68/features/regimes.py` | Labels each period with its regime: pre_crisis, gas_crisis, iberian_exception, normalization. |
| `src/pce68/features/metrics.py` | Daily spread and time-weighted statistics. |
| `src/pce68/dispatch/battery.py` | Arbitrage of 1 MW of battery, price taker, one cycle per day. |
| `src/pce68/market/curves.py` | Clearing price of a period's curves and calibration to the historical price. |
| `src/pce68/dispatch/fleet.py` | Dispatch of a fleet of X GW that moves the prices it trades at, up to the competitive equilibrium. |
| `src/pce68/market/impact.py` | Prices of each day with X GW of storage, and the impact y = pX / p0. |

The assumptions behind each calculation are documented at the top of its module.
