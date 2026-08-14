"""Typed application configuration.

Values load from the process environment, falling back to a `.env` file in
the working directory, then the defaults below. See `.env.example` for the
full documented variable list.
"""

from functools import lru_cache

from pydantic import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Instantiate via `get_settings()`, not directly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # An unfilled-in optional var (e.g. FRONTEND_ORIGIN=) should fall back
        # to its default rather than fail validation as an empty string.
        env_ignore_empty=True,
    )

    # Secrets — no default; required once the sessions that use them land.
    tmdb_api_key: str | None = None
    api_key: str | None = None
    cloudflare_tunnel_token: str | None = None

    # TMDB ingestion — conservative default, not a documented hard API limit.
    tmdb_requests_per_second: float = 4.0

    # Frontend <-> backend wiring
    frontend_origin: AnyUrl | None = None
    api_base_url: AnyUrl = AnyUrl("http://api:8000")

    # MongoDB
    mongodb_uri: AnyUrl = AnyUrl("mongodb://localhost:27017/?directConnection=true")
    mongodb_db: str = "movie_recs"

    # vLLM
    vllm_base_url: AnyUrl = AnyUrl("http://vllm:8000/v1")
    vllm_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Embedding service
    embed_base_url: AnyUrl = AnyUrl("http://embedding:8080")
    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    clip_model: str = "ViT-B-32"
    embed_batch_size: int = 64
    # `None` lets sentence-transformers pick (CUDA when available); set to
    # "cpu"/"cuda" to force. Also used by the offline catalog embed job.
    embed_device: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached `Settings` instance."""
    return Settings()
