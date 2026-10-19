.PHONY: data excel excel-demo test lint

data:
	uv run python scripts/01_download_omie.py

excel:
	uv run python scripts/04_export_excel.py

excel-demo:
	uv run python scripts/04_export_excel.py --demo

test:
	uv run pytest -q

lint:
	uv run ruff check .
