"""Tests for movie_recs.main."""

from movie_recs import __version__
from movie_recs.main import health


def test_health_returns_pinned_version_string() -> None:
    assert health() == f"movie-recs v{__version__} OK"
    assert __version__ == "0.1.0"
