.PHONY: sync sync-ml test lint fmt embed-text

sync:
	uv sync --frozen

# Adds the `ml` group (sentence-transformers + ~5 GB of CUDA torch wheels);
# needed to run the embedding service or the offline embed jobs locally.
sync-ml:
	uv sync --frozen --group ml

test:
	uv run pytest -m "not integration"

lint:
	uv run ruff check .
	uv run mypy

fmt:
	uv run ruff format .
	uv run ruff check --fix .

# Offline: embed the catalog's text and build/await the `text_vec` index.
# Requires `make sync-ml` and a running mongodb.
embed-text:
	uv run python -m movie_recs.embeddings
