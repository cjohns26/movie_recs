"""Non-vector Mongo index bootstrap.

Vector search indexes (`text_vec`, `poster_vec` on `movies`) are added in
Sessions 4-5, once those embedding fields exist.
"""

import logging

from movie_recs.db.collections import (
    movies_collection,
    ratings_collection,
    tmdb_cache_collection,
)

logger = logging.getLogger(__name__)


def ensure_indexes() -> None:
    """Create the indexes documented in plan.md's Data Model.

    `create_index` is idempotent — a no-op if the index already exists with
    the same spec.
    """
    movies_collection().create_index("tmdbId")
    movies_collection().create_index("genres")
    ratings_collection().create_index("userId")
    ratings_collection().create_index("movieId")
    ratings_collection().create_index("timestamp")
    tmdb_cache_collection().create_index("tmdbId", unique=True)
    logger.info("Mongo indexes ensured")
