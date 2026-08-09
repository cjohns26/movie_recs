"""Tests for movie_recs.config.Settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from movie_recs.config import Settings


def test_settings_loads_values_from_env_file(tmp_path: Path) -> None:
    """Values present in a .env file override the built-in defaults."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TMDB_API_KEY=test-tmdb-key\n"
        "MONGODB_DB=movie_recs_test\n"
        "VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct\n"
    )

    settings = Settings(_env_file=str(env_file))

    assert settings.tmdb_api_key == "test-tmdb-key"
    assert settings.mongodb_db == "movie_recs_test"
    assert settings.vllm_model == "Qwen/Qwen2.5-3B-Instruct"


def test_settings_applies_defaults_when_env_file_is_empty(tmp_path: Path) -> None:
    """With no matching env vars, documented defaults are used."""
    env_file = tmp_path / ".env"
    env_file.write_text("")

    settings = Settings(_env_file=str(env_file))

    assert settings.tmdb_api_key is None
    assert settings.mongodb_db == "movie_recs"
    assert str(settings.mongodb_uri) == "mongodb://localhost:27017/?directConnection=true"
    assert settings.vllm_model == "meta-llama/Llama-3.2-3B-Instruct"


def test_settings_rejects_malformed_url(tmp_path: Path) -> None:
    """A malformed URL-typed value raises instead of silently loading."""
    env_file = tmp_path / ".env"
    env_file.write_text("API_BASE_URL=not-a-url\n")

    with pytest.raises(ValidationError):
        Settings(_env_file=str(env_file))
