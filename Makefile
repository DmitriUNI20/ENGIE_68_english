.PHONY: data curves excel excel-demo fleet fleet-demo test lint

data:
	uv run python scripts/01_download_omie.py

curves:
	uv run python scripts/02_download_curves.py

excel:
	uv run python scripts/04_export_excel.py

excel-demo:
	uv run python scripts/04_export_excel.py --demo

fleet:
	uv run python scripts/05_fleet.py --regime normalization

fleet-demo:
	uv run python scripts/05_fleet.py --demo

test:
	uv run pytest -q

lint:
	uv run ruff check .
