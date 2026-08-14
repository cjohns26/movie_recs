"""Atlas `$vectorSearch` index definitions + readiness polling.

Separate from `db.indexes` because these are `mongot` search indexes, not
regular `mongod` indexes: they're created through `createSearchIndexes` and —
unlike `create_index` — they build **asynchronously**. Querying one before it
reports `queryable` returns an error or an empty result set (Risk #6), so
every bootstrap path must `wait_for_vector_index` before searching.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

logger = logging.getLogger(__name__)

TEXT_EMBEDDING_FIELD = "text_embedding"
TEXT_VECTOR_INDEX = "text_vec"

DEFAULT_READY_TIMEOUT_S = 300.0
DEFAULT_POLL_INTERVAL_S = 2.0


@dataclass(frozen=True)
class VectorIndexSpec:
    """One `type: "vectorSearch"` index over a single embedding field."""

    name: str
    path: str
    dimensions: int
    similarity: str = "cosine"

    def to_model(self) -> SearchIndexModel:
        return SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": self.path,
                        "numDimensions": self.dimensions,
                        "similarity": self.similarity,
                    }
                ]
            },
            name=self.name,
            type="vectorSearch",
        )


def text_vector_spec() -> VectorIndexSpec:
    """The `movies.text_embedding` index (384-dim, cosine)."""
    from movie_recs.embeddings.text import TEXT_EMBED_DIM

    return VectorIndexSpec(
        name=TEXT_VECTOR_INDEX, path=TEXT_EMBEDDING_FIELD, dimensions=TEXT_EMBED_DIM
    )


def vector_index_info(collection: Collection[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Return the search index's status document, or None if it doesn't exist."""
    for index in collection.list_search_indexes(name):
        return dict(index)
    return None


def ensure_vector_index(collection: Collection[dict[str, Any]], spec: VectorIndexSpec) -> bool:
    """Create `spec` if absent. Returns True if it was created by this call.

    Idempotent: an existing index of the same name is left untouched (Atlas
    rejects a duplicate name, and re-creating would drop and rebuild it).
    """
    if vector_index_info(collection, spec.name) is not None:
        logger.info("Vector index %r already exists on %s", spec.name, collection.name)
        return False
    try:
        collection.create_search_index(model=spec.to_model())
    except OperationFailure:
        # Lost a race with a concurrent bootstrap: fine if it now exists.
        if vector_index_info(collection, spec.name) is None:
            raise
        logger.info("Vector index %r created concurrently", spec.name)
        return False
    logger.info("Created vector index %r on %s.%s", spec.name, collection.name, spec.path)
    return True


def wait_for_vector_index(
    collection: Collection[dict[str, Any]],
    name: str,
    *,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, Any]:
    """Block until the index reports `queryable`, then return its status doc.

    A freshly created index takes a moment to appear in `list_search_indexes`
    at all, so "not listed yet" counts as not-ready rather than an error until
    the timeout expires.

    Note this only covers the *index*: `mongot` also indexes subsequent
    document writes asynchronously, so a movie embedded after the index went
    queryable may take a few more seconds to turn up in `$vectorSearch` hits.

    Raises:
        TimeoutError: if it isn't queryable within `timeout_s`.
        RuntimeError: if the index build failed.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        info = vector_index_info(collection, name)
        if info is not None:
            if info.get("status") == "FAILED":
                raise RuntimeError(f"vector index {name!r} build failed: {info}")
            if info.get("queryable"):
                logger.info("Vector index %r is queryable (status=%s)", name, info.get("status"))
                return info
        if time.monotonic() >= deadline:
            status = "missing" if info is None else repr(info.get("status"))
            raise TimeoutError(
                f"vector index {name!r} on {collection.name} not queryable after "
                f"{timeout_s:.0f}s (status={status})"
            )
        time.sleep(poll_interval_s)
