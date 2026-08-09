"""Tests for movie_recs.ingest.join."""

from pathlib import Path

import pandas as pd

from movie_recs.ingest.join import join_movies_links, merge_tmdb_metadata
from movie_recs.ingest.movielens import load_links, load_movies

FIXTURES = Path(__file__).parent.parent / "fixtures" / "ml-latest-small"


def test_join_movies_links_maps_known_movie_to_correct_tmdb_id() -> None:
    movies_df = load_movies(FIXTURES / "movies.csv")
    links_df = load_links(FIXTURES / "links.csv")

    joined = join_movies_links(movies_df, links_df)

    row = joined.loc[joined["movieId"] == 1].iloc[0]
    assert row["tmdbId"] == 862


def test_join_movies_links_handles_unmatched_rows_without_crashing() -> None:
    """movieId 3 and 5 have no row in links.csv at all."""
    movies_df = load_movies(FIXTURES / "movies.csv")
    links_df = load_links(FIXTURES / "links.csv")

    joined = join_movies_links(movies_df, links_df)

    assert len(joined) == len(movies_df)
    row3 = joined.loc[joined["movieId"] == 3].iloc[0]
    assert pd.isna(row3["tmdbId"])


def test_merge_tmdb_metadata_overlays_payload_fields() -> None:
    row = {"movieId": 1, "title": "Toy Story (1995)"}
    payload = {
        "overview": "A cowboy doll...",
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "popularity": 91.2,
        "vote_average": 8.3,
        "unused_field": "ignored",
    }

    merged = merge_tmdb_metadata(row, payload)

    assert merged["overview"] == "A cowboy doll..."
    assert merged["poster_path"] == "/poster.jpg"
    assert merged["vote_average"] == 8.3
    assert "unused_field" not in merged
    assert merged["title"] == "Toy Story (1995)"  # original fields preserved


def test_merge_tmdb_metadata_with_no_payload_leaves_row_unchanged() -> None:
    row = {"movieId": 3, "title": "Grumpier Old Men (1995)"}

    merged = merge_tmdb_metadata(row, None)

    assert merged == row
