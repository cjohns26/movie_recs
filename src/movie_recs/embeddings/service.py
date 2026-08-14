"""FastAPI embedding microservice (GPU).

Serves query-time embeddings for the recommendation API; catalog vectors are
precomputed offline by `movie_recs.embeddings.run`. Session 5 adds
`/embed/image` (CLIP) alongside `/embed/text`.

Run locally:  uv run uvicorn movie_recs.embeddings.service:app --port 8080
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from movie_recs.embeddings.text import TEXT_EMBED_DIM, TextEmbedder

logger = logging.getLogger(__name__)

# Bounded so one request can't pin the GPU for an unbounded batch.
MAX_BATCH = 256


@lru_cache
def get_text_embedder() -> TextEmbedder:
    """Process-wide text embedder. Overridden in tests via `dependency_overrides`."""
    from movie_recs.config import get_settings

    return TextEmbedder(device=get_settings().embed_device)


class EmbedTextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class EmbedTextResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]


class HealthResponse(BaseModel):
    status: str
    text_model: str
    text_dim: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the model at startup so the first request isn't a cold load.

    Resolves through `dependency_overrides` so a test client that stubs the
    embedder doesn't download the real model on startup.
    """
    provider = app.dependency_overrides.get(get_text_embedder, get_text_embedder)
    try:
        provider().load()
    except Exception:
        # A failed warmup shouldn't stop the service from starting — /health
        # stays reachable and the next request retries the load.
        logger.exception("Text model warmup failed; will retry on first request")
    yield


app = FastAPI(title="movie-recs embedding service", lifespan=lifespan)


@app.get("/health")
def health(embedder: Annotated[TextEmbedder, Depends(get_text_embedder)]) -> HealthResponse:
    return HealthResponse(status="ok", text_model=embedder.model_name, text_dim=TEXT_EMBED_DIM)


@app.post("/embed/text")
def embed_text(
    request: EmbedTextRequest,
    embedder: Annotated[TextEmbedder, Depends(get_text_embedder)],
) -> EmbedTextResponse:
    """Embed a batch of strings, preserving input order."""
    vectors = embedder.encode(request.texts)
    return EmbedTextResponse(model=embedder.model_name, dim=TEXT_EMBED_DIM, embeddings=vectors)
