"""Tests for movie_recs.db.vector_indexes.

The search-index lifecycle is faked here (no Mongo): these pin the *polling*
contract, which is what stops a bootstrap from querying an index that isn't
queryable yet (plan.md Risk #6).
"""

from typing import Any

import pytest

from movie_recs.db.vector_indexes import (
    TEXT_EMBEDDING_FIELD,
    TEXT_VECTOR_INDEX,
    ensure_vector_index,
    text_vector_spec,
    wait_for_vector_index,
)
from movie_recs.embeddings.text import TEXT_EMBED_DIM


class FakeSearchIndexCollection:
    """Replays a scripted sequence of `list_search_indexes` responses."""

    name = "movies"

    def __init__(self, states: list[dict[str, Any] | None]) -> None:
        self.states = states
        self.created: list[Any] = []

    def list_search_indexes(self, name: str) -> list[dict[str, Any]]:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return [] if state is None else [{"name": name, **state}]

    def create_search_index(self, model: Any) -> str:
        self.created.append(model)
        return TEXT_VECTOR_INDEX


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("movie_recs.db.vector_indexes.time.sleep", lambda _: None)


def test_text_spec_matches_the_stored_embedding_field_and_dimension() -> None:
    """The index declares `numDimensions` up front, so it must agree with the
    model's output dim or every insert is rejected."""
    spec = text_vector_spec()

    assert (spec.name, spec.path, spec.dimensions) == (
        TEXT_VECTOR_INDEX,
        TEXT_EMBEDDING_FIELD,
        TEXT_EMBED_DIM,
    )
    definition = spec.to_model().document["definition"]
    assert definition["fields"][0]["similarity"] == "cosine"


def test_ensure_vector_index_creates_when_absent() -> None:
    collection = FakeSearchIndexCollection([None])

    created = ensure_vector_index(collection, text_vector_spec())  # type: ignore[arg-type]

    assert created is True
    assert len(collection.created) == 1


def test_ensure_vector_index_is_a_no_op_when_present() -> None:
    """Re-creating an existing Atlas search index would drop and rebuild it —
    an idempotent bootstrap must leave it alone."""
    collection = FakeSearchIndexCollection([{"queryable": True, "status": "READY"}])

    created = ensure_vector_index(collection, text_vector_spec())  # type: ignore[arg-type]

    assert created is False
    assert collection.created == []


def test_wait_returns_once_the_index_becomes_queryable() -> None:
    collection = FakeSearchIndexCollection(
        [
            None,  # not listed yet right after creation
            {"queryable": False, "status": "PENDING"},
            {"queryable": True, "status": "READY"},
        ]
    )

    info = wait_for_vector_index(collection, TEXT_VECTOR_INDEX, timeout_s=10)  # type: ignore[arg-type]

    assert info["status"] == "READY"


def test_wait_times_out_when_the_index_never_appears() -> None:
    """Guards the failure mode where a typo'd name would otherwise hang or
    look like an empty result set."""
    collection = FakeSearchIndexCollection([None])

    with pytest.raises(TimeoutError, match="missing"):
        wait_for_vector_index(collection, TEXT_VECTOR_INDEX, timeout_s=0)  # type: ignore[arg-type]


def test_wait_raises_immediately_on_a_failed_build() -> None:
    collection = FakeSearchIndexCollection([{"queryable": False, "status": "FAILED"}])

    with pytest.raises(RuntimeError, match="build failed"):
        wait_for_vector_index(collection, TEXT_VECTOR_INDEX, timeout_s=10)  # type: ignore[arg-type]
