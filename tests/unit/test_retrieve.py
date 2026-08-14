"""Tests for movie_recs.recsys.retrieve.

The `$vectorSearch` pipeline is asserted structurally and the collection is
faked, so the retrieval contract is pinned without a running Mongo. The
end-to-end "are these actually good neighbors" question is the integration
test's job.
"""

from typing import Any

import pytest

from movie_recs.db.vector_indexes import TEXT_EMBEDDING_FIELD, TEXT_VECTOR_INDEX
from movie_recs.recsys.retrieve import (
    MIN_NUM_CANDIDATES,
    NUM_CANDIDATES_MULTIPLIER,
    build_vector_search_pipeline,
    semantic_neighbors,
    similar_by_text,
)


class FakeCollection:
    """Records the pipeline it was given and replays canned hits."""

    def __init__(self, hits: list[dict[str, Any]], doc: dict[str, Any] | None = None) -> None:
        self.hits = hits
        self.doc = doc
        self.pipelines: list[list[dict[str, Any]]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.pipelines.append(pipeline)
        limit = pipeline[0]["$vectorSearch"]["limit"]
        return self.hits[:limit]

    def find_one(self, query: dict[str, Any], projection: dict[str, int]) -> dict[str, Any] | None:
        return self.doc


def test_pipeline_targets_the_text_index_and_projects_the_similarity_score() -> None:
    pipeline = build_vector_search_pipeline(
        [0.1, 0.2], index=TEXT_VECTOR_INDEX, path=TEXT_EMBEDDING_FIELD, limit=5
    )

    stage = pipeline[0]["$vectorSearch"]
    assert stage["index"] == TEXT_VECTOR_INDEX
    assert stage["path"] == TEXT_EMBEDDING_FIELD
    assert stage["queryVector"] == [0.1, 0.2]
    assert stage["limit"] == 5
    assert pipeline[1]["$project"]["score"] == {"$meta": "vectorSearchScore"}


def test_num_candidates_scales_with_limit_but_never_below_the_floor() -> None:
    """`numCandidates` is the ANN search width — too small and recall collapses."""
    narrow = build_vector_search_pipeline([0.0], index="i", path="p", limit=2)
    wide = build_vector_search_pipeline([0.0], index="i", path="p", limit=50)

    assert narrow[0]["$vectorSearch"]["numCandidates"] == MIN_NUM_CANDIDATES
    assert wide[0]["$vectorSearch"]["numCandidates"] == 50 * NUM_CANDIDATES_MULTIPLIER


def test_pipeline_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        build_vector_search_pipeline([0.0], index="i", path="p", limit=0)


def test_semantic_neighbors_maps_hits_to_candidates_in_rank_order() -> None:
    collection = FakeCollection(
        [{"_id": 7, "score": 0.9}, {"_id": 8, "score": 0.8}, {"_id": 9, "score": 0.7}]
    )

    candidates = semantic_neighbors([0.0], k=2, collection=collection)  # type: ignore[arg-type]

    assert [(c.movieId, c.score) for c in candidates] == [(7, 0.9), (8, 0.8)]
    assert {c.source for c in candidates} == {"text"}


def test_exclusions_are_over_fetched_so_k_results_still_come_back() -> None:
    """Excluding client-side would silently return k-1 results if the query
    didn't ask for extra rows to absorb the exclusions."""
    collection = FakeCollection(
        [{"_id": 1, "score": 0.99}, {"_id": 7, "score": 0.9}, {"_id": 8, "score": 0.8}]
    )

    candidates = semantic_neighbors([0.0], k=2, exclude_ids=[1], collection=collection)  # type: ignore[arg-type]

    assert collection.pipelines[0][0]["$vectorSearch"]["limit"] == 3
    assert [c.movieId for c in candidates] == [7, 8]


def test_similar_by_text_uses_the_stored_vector_and_excludes_the_seed() -> None:
    collection = FakeCollection(
        hits=[{"_id": 1, "score": 1.0}, {"_id": 2, "score": 0.9}],
        doc={"_id": 1, TEXT_EMBEDDING_FIELD: [0.3, 0.4]},
    )

    candidates = similar_by_text(1, k=1, collection=collection)  # type: ignore[arg-type]

    assert collection.pipelines[0][0]["$vectorSearch"]["queryVector"] == [0.3, 0.4]
    assert [c.movieId for c in candidates] == [2]


def test_similar_by_text_reports_an_unembedded_movie_rather_than_returning_junk() -> None:
    """Until the embed job has run (or for a movie it skipped), the caller needs
    an explicit failure, not an empty list that looks like "no similar movies"."""
    collection = FakeCollection(hits=[], doc={"_id": 1})

    with pytest.raises(LookupError, match=TEXT_EMBEDDING_FIELD):
        similar_by_text(1, collection=collection)  # type: ignore[arg-type]


def test_similar_by_text_raises_for_an_unknown_movie() -> None:
    collection = FakeCollection(hits=[], doc=None)

    with pytest.raises(LookupError, match="not found"):
        similar_by_text(999, collection=collection)  # type: ignore[arg-type]
