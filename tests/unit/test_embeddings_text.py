"""Tests for movie_recs.embeddings.text's document builder.

Pure-logic only — no model is loaded, so these run in CI's torch-free env.
"""

import pytest

from movie_recs.embeddings.text import MAX_TAGS, build_text_document

FULL_MOVIE = {
    "_id": 1,
    "title": "Toy Story (1995)",
    "year": 1995,
    "genres": ["Adventure", "Animation", "Children"],
    "overview": "Woody the cowboy doll is threatened by a new spaceman figure.",
    "tags": ["pixar", "fun"],
}


def test_full_movie_includes_every_section() -> None:
    document = build_text_document(FULL_MOVIE)

    assert document.splitlines() == [
        "Toy Story (1995)",
        "Genres: Adventure, Animation, Children",
        "Woody the cowboy doll is threatened by a new spaceman figure.",
        "Tags: pixar, fun",
    ]


def test_missing_overview_falls_back_to_title_genres_tags() -> None:
    """122 of the 9,742 ingested movies have no TMDB overview. They must still
    produce a meaningful embedding input, not an empty string (plan.md Session 4).
    """
    movie = {**FULL_MOVIE, "overview": None}

    document = build_text_document(movie)

    assert "Woody" not in document
    assert document.startswith("Toy Story (1995)")
    assert "Genres: Adventure, Animation, Children" in document
    assert "Tags: pixar, fun" in document


def test_title_only_movie_still_yields_non_empty_document() -> None:
    """The worst case — no TMDB match, no genres, no tags — is still embeddable."""
    document = build_text_document(
        {"_id": 42, "title": "Untitled (2001)", "genres": [], "tags": []}
    )

    assert document == "Untitled (2001)"


def test_blank_overview_is_treated_as_missing() -> None:
    document = build_text_document({**FULL_MOVIE, "overview": "   "})

    assert document.splitlines() == [
        "Toy Story (1995)",
        "Genres: Adventure, Animation, Children",
        "Tags: pixar, fun",
    ]


def test_year_is_appended_only_when_absent_from_the_title() -> None:
    """MovieLens embeds the year in the title, so the ingested `year` column is
    usually redundant — appending it unconditionally would double it."""
    assert build_text_document({"title": "Heat", "year": 1995}) == "Heat (1995)"
    assert build_text_document({"title": "Heat (1995)", "year": 1995}) == "Heat (1995)"


def test_tags_are_deduplicated_case_insensitively_and_capped() -> None:
    movie = {
        "title": "Tagged",
        "tags": ["Pixar", "pixar", "  fun  ", ""] + [f"tag{i}" for i in range(MAX_TAGS)],
        "genres": ["Comedy", "comedy"],
    }

    document = build_text_document(movie)

    tags = document.splitlines()[-1].removeprefix("Tags: ").split(", ")
    assert tags[:2] == ["Pixar", "fun"]
    assert len(tags) == MAX_TAGS
    assert "Genres: Comedy" in document


def test_missing_title_raises() -> None:
    with pytest.raises(ValueError, match="no title"):
        build_text_document({"_id": 7, "title": "", "overview": "an overview"})
