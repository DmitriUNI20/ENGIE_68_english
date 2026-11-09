.PHONY: all data curves impact excel excel-demo fleet plot test lint

all:
	uv run python scripts/run_all.py

data:
	uv run python scripts/01_download_omie.py

curves:
	uv run python scripts/02_download_curves.py

impact:
	uv run python scripts/03_simulate_impact.py

excel:
	uv run python scripts/04_export_excel.py

excel-demo:
	uv run python scripts/04_export_excel.py --demo

fleet:
	uv run python scripts/05_fleet.py

plot:
	uv run python scripts/06_plot_fleet.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
