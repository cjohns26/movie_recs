.PHONY: sync test lint fmt

sync:
	uv sync --frozen

test:
	uv run pytest -m "not integration"

lint:
	uv run ruff check .
	uv run mypy

fmt:
	uv run ruff format .
	uv run ruff check --fix .
