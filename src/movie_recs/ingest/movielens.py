"""Parse MovieLens `ml-latest-small` CSVs (movies, ratings, links, tags)."""

import re
from pathlib import Path
from typing import cast

import pandas as pd

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_NO_GENRES = "(no genres listed)"


def _parse_year(title: str) -> int | None:
    """Extract the trailing "(YYYY)" year MovieLens embeds in the title."""
    match = _YEAR_RE.search(title)
    return int(match.group(1)) if match else None


def _parse_genres(genres: str) -> list[str]:
    """Split MovieLens's pipe-delimited genre string into a list."""
    if genres == _NO_GENRES:
        return []
    return genres.split("|")


def load_movies(path: Path) -> pd.DataFrame:
    """Load movies.csv, adding parsed `year` and list-typed `genres` columns."""
    df = pd.read_csv(path)
    df["year"] = df["title"].apply(_parse_year)
    df["genres"] = df["genres"].apply(_parse_genres)
    return df


def load_ratings(path: Path) -> pd.DataFrame:
    """Load ratings.csv (userId, movieId, rating, timestamp) unchanged."""
    return pd.read_csv(path)


def load_links(path: Path) -> pd.DataFrame:
    """Load links.csv. `tmdbId` is nullable (some movies have no TMDB match).

    `imdbId` is read as text and zero-padded to GroupLens's documented
    7-digit convention (https://www.imdb.com/title/tt{imdbId}/), since
    pandas would otherwise parse it as an int and drop leading zeros.
    """
    df = pd.read_csv(path, dtype={"imdbId": "string", "tmdbId": "Int64"})
    df["imdbId"] = df["imdbId"].str.zfill(7)
    return df


def load_tags(path: Path) -> pd.DataFrame:
    """Load tags.csv (userId, movieId, tag, timestamp) unchanged."""
    return pd.read_csv(path)


def tags_by_movie(tags_df: pd.DataFrame) -> dict[int, list[str]]:
    """Group free-text tags by movieId, deduplicated and order-preserving."""
    out: dict[int, list[str]] = {}
    for movie_id, group in tags_df.groupby("movieId"):
        seen: list[str] = []
        for tag in group["tag"]:
            if tag not in seen:
                seen.append(tag)
        # groupby's key type is a broad Hashable union per pandas-stubs; the
        # "movieId" column is always numeric at runtime.
        out[int(cast(int, movie_id))] = seen
    return out
