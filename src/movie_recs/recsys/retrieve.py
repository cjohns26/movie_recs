"""Candidate generation over MongoDB `$vectorSearch`.

This is the retrieval half of the two-stage pipeline: it returns *candidates*
with a raw similarity score, not a final ranking — Session 6's reranker blends
these with CF, popularity and recency. Session 5 adds the poster/visual and
hybrid paths here.

Vector retrieval is also the only way to reach the ~4,500 movies with no ALS
factor (Risk #16) and any newly-inserted movie with zero ratings (item
cold-start): similarity needs the embedding, not the rating history.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pymongo.collection import Collection

from movie_recs.db.collections import movies_collection
from movie_recs.db.vector_indexes import TEXT_EMBEDDING_FIELD, TEXT_VECTOR_INDEX

logger = logging.getLogger(__name__)

# `numCandidates` is the ANN search width: higher = better recall, slower.
# Atlas guidance is 10-20x the requested `limit`, with a floor so small k
# values still explore enough of the graph.
NUM_CANDIDATES_MULTIPLIER = 10
MIN_NUM_CANDIDATES = 100


@dataclass(frozen=True)
class Candidate:
    """One retrieved movie. `score` is index-relative — comparable within a
    source, not across sources (that's the reranker's job)."""

    movieId: int
    score: float
    source: str


def build_vector_search_pipeline(
    query_vector: Sequence[float],
    *,
    index: str,
    path: str,
    limit: int,
    num_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Build the `$vectorSearch` aggregation pipeline (pure — no I/O)."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return [
        {
            "$vectorSearch": {
                "index": index,
                "path": path,
                "queryVector": [float(x) for x in query_vector],
                "numCandidates": num_candidates
                or max(MIN_NUM_CANDIDATES, limit * NUM_CANDIDATES_MULTIPLIER),
                "limit": limit,
            }
        },
        {"$project": {"_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]


def semantic_neighbors(
    query_vector: Sequence[float],
    *,
    k: int = 10,
    exclude_ids: Iterable[int] = (),
    collection: Collection[dict[str, Any]] | None = None,
) -> list[Candidate]:
    """Return the k nearest movies to `query_vector` by text embedding.

    Exclusions are applied client-side on an over-fetched result set: filtering
    inside `$vectorSearch` would require `_id` to be declared as a filter field
    on the index, and the exclusion sets here (a session's seen items) are
    small relative to k.
    """
    excluded = {int(i) for i in exclude_ids}
    coll = collection if collection is not None else movies_collection()
    pipeline = build_vector_search_pipeline(
        query_vector,
        index=TEXT_VECTOR_INDEX,
        path=TEXT_EMBEDDING_FIELD,
        limit=k + len(excluded),
    )
    candidates = [
        Candidate(movieId=int(doc["_id"]), score=float(doc["score"]), source="text")
        for doc in coll.aggregate(pipeline)
        if int(doc["_id"]) not in excluded
    ]
    return candidates[:k]


def similar_by_text(
    movie_id: int,
    *,
    k: int = 10,
    collection: Collection[dict[str, Any]] | None = None,
) -> list[Candidate]:
    """Return the k movies most similar to `movie_id` by text embedding.

    Raises:
        LookupError: if the movie doesn't exist or hasn't been embedded yet.
    """
    coll = collection if collection is not None else movies_collection()
    doc = coll.find_one({"_id": movie_id}, {TEXT_EMBEDDING_FIELD: 1})
    if doc is None:
        raise LookupError(f"movie {movie_id} not found")
    vector = doc.get(TEXT_EMBEDDING_FIELD)
    if not vector:
        raise LookupError(f"movie {movie_id} has no {TEXT_EMBEDDING_FIELD}; run the embed job")
    return semantic_neighbors(vector, k=k, exclude_ids=[movie_id], collection=coll)
