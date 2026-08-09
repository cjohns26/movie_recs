"""Join MovieLens movieId -> tmdbId, and overlay TMDB metadata onto a movie record."""

from typing import Any

import pandas as pd


def join_movies_links(movies_df: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join movies to links on movieId.

    Movies with no matching links.csv row, or a null tmdbId within it, keep
    tmdbId as NA and are still ingested — content/poster similarity (item
    cold-start, Session 4/5) doesn't require a TMDB match.
    """
    return movies_df.merge(links_df[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")


def merge_tmdb_metadata(row: dict[str, Any], tmdb_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay TMDB fields onto a movie record.

    A missing payload (no tmdbId match, or a 404 from TMDB) leaves those
    fields unset rather than failing the row.
    """
    if tmdb_payload is None:
        return row
    return {
        **row,
        "overview": tmdb_payload.get("overview"),
        "poster_path": tmdb_payload.get("poster_path"),
        "backdrop_path": tmdb_payload.get("backdrop_path"),
        "popularity": tmdb_payload.get("popularity"),
        "vote_average": tmdb_payload.get("vote_average"),
    }
