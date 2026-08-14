"""Offline job: embed the catalog's text and build the `text_vec` index.

Runs in-process against the GPU (same image as the embedding service) rather
than over HTTP — this is a batch job over the whole catalog, and the service's
per-request batch cap exists for online traffic, not for this.

Usage: uv run python -m movie_recs.embeddings [--force] [--limit N] [--skip-index]
"""

import argparse
import logging
from typing import Any

from pymongo import UpdateOne

from movie_recs.config import get_settings
from movie_recs.db.collections import movies_collection
from movie_recs.db.vector_indexes import (
    TEXT_EMBEDDING_FIELD,
    ensure_vector_index,
    text_vector_spec,
    wait_for_vector_index,
)
from movie_recs.embeddings.text import TextEmbedder, build_text_document

logger = logging.getLogger(__name__)

# Fields the document builder needs — projected so a 9.7k-movie scan doesn't
# drag the (already stored) embeddings back over the wire.
_PROJECTION = {"title": 1, "year": 1, "genres": 1, "overview": 1, "tags": 1}


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def embed_catalog(
    embedder: TextEmbedder,
    *,
    force: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
) -> dict[str, int]:
    """Embed movies' text and store `text_embedding` on each document.

    Idempotent: without `--force` only movies missing an embedding are
    processed, so a re-run after a partial failure resumes rather than redoes.

    Args:
        force: re-embed movies that already have a `text_embedding`.
        limit: stop after this many movies (for smoke runs).
        batch_size: encoder batch size; defaults to `EMBED_BATCH_SIZE`.
    """
    collection = movies_collection()
    size = batch_size or get_settings().embed_batch_size
    query: dict[str, Any] = {} if force else {TEXT_EMBEDDING_FIELD: {"$exists": False}}

    cursor = collection.find(query, _PROJECTION).sort("_id", 1)
    if limit is not None:
        cursor = cursor.limit(limit)
    pending = list(cursor)

    total = collection.count_documents({})
    logger.info("Embedding %d of %d movies (force=%s)", len(pending), total, force)

    embedded = 0
    for batch in _batches(pending, size):
        documents = [build_text_document(movie) for movie in batch]
        vectors = embedder.encode(documents, batch_size=size)
        collection.bulk_write(
            [
                UpdateOne({"_id": movie["_id"]}, {"$set": {TEXT_EMBEDDING_FIELD: vector}})
                for movie, vector in zip(batch, vectors, strict=True)
            ]
        )
        embedded += len(batch)
        logger.info("Embedded %d/%d", embedded, len(pending))

    counts = {
        "embedded": embedded,
        "skipped": total - embedded if not force else 0,
        "total": total,
        "with_embedding": collection.count_documents({TEXT_EMBEDDING_FIELD: {"$exists": True}}),
    }
    logger.info("Catalog text embedding complete: %s", counts)
    return counts


def build_text_index(*, wait: bool = True) -> None:
    """Create `text_vec` if absent and (by default) wait until it's queryable."""
    collection = movies_collection()
    spec = text_vector_spec()
    ensure_vector_index(collection, spec)
    if wait:
        wait_for_vector_index(collection, spec.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # httpx logs every Hugging Face model-download request at INFO; the job's
    # own progress lines are the signal here.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Embed movie text into MongoDB + build text_vec")
    parser.add_argument(
        "--force", action="store_true", help="Re-embed movies that already have a vector"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N movies")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--skip-index", action="store_true", help="Embed only; don't create/await text_vec"
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder = TextEmbedder(device=settings.embed_device)
    embed_catalog(embedder, force=args.force, limit=args.limit, batch_size=args.batch_size)
    if not args.skip_index:
        build_text_index()


if __name__ == "__main__":
    main()
