"""Text embeddings: catalog document builder + sentence-transformers wrapper.

`build_text_document` is deliberately pure (dict in, str out) so the document
format is testable without loading a model — 122 of the 9,742 ingested movies
have no TMDB `overview`, and those must still produce a meaningful embedding
input from `title + genres + tags` rather than an empty string.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from movie_recs.config import get_settings

logger = logging.getLogger(__name__)

# Output dimensionality of TEXT_EMBED_MODEL (BAAI/bge-small-en-v1.5).
# Pinned here because the `text_vec` vector index must declare it up front;
# `TextEmbedder` asserts the loaded model agrees.
TEXT_EMBED_DIM = 384

# Tags are crowd-sourced and long-tailed; cap them so one heavily tagged movie
# doesn't dominate its own document (and blow past the model's 512-token window).
MAX_TAGS = 20


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _dedupe(values: Sequence[Any]) -> list[str]:
    """Case-insensitive dedupe, preserving first-seen order and casing."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = _clean(value)
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        out.append(cleaned)
    return out


def build_text_document(movie: Mapping[str, Any]) -> str:
    """Render a `movies` document into the string that gets embedded.

    Every section is optional except the title: a movie with no `overview`
    (or no genres, or no tags) still yields a non-empty document. Sections are
    labelled so the model sees why a term is present.

    Raises:
        ValueError: if the movie has no usable title.
    """
    title = _clean(movie.get("title"))
    if not title:
        raise ValueError(f"movie {movie.get('_id')!r} has no title to embed")

    # MovieLens titles usually already carry the year ("Toy Story (1995)"),
    # so only append it when the parsed year isn't already in the title.
    year = movie.get("year")
    header = title if year is None or f"({int(year)})" in title else f"{title} ({int(year)})"

    lines = [header]
    genres = _dedupe(movie.get("genres") or [])
    if genres:
        lines.append(f"Genres: {', '.join(genres)}")
    overview = _clean(movie.get("overview"))
    if overview:
        lines.append(overview)
    tags = _dedupe(movie.get("tags") or [])[:MAX_TAGS]
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    return "\n".join(lines)


class TextEmbedder:
    """Lazily-loaded sentence-transformers encoder producing unit vectors.

    The model import and download happen on first use, not at construction, so
    importing this module stays cheap and torch-free environments (CI, the API
    container) can import the service module without the `ml` dependency group.
    """

    def __init__(self, model_name: str | None = None, *, device: str | None = None) -> None:
        self.model_name = model_name or get_settings().text_embed_model
        self.device = device
        self._model: Any | None = None

    def load(self) -> None:
        """Load the model (idempotent). Raises if its dim isn't TEXT_EMBED_DIM."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading text embedding model %s (device=%s)", self.model_name, self.device)
        model = SentenceTransformer(self.model_name, device=self.device)
        # `get_sentence_embedding_dimension` is the deprecated spelling in
        # sentence-transformers >= 5.7.
        dim = int(model.get_embedding_dimension())
        if dim != TEXT_EMBED_DIM:
            raise ValueError(
                f"{self.model_name} produces {dim}-dim vectors, but the text_vec index "
                f"and stored embeddings are pinned to {TEXT_EMBED_DIM}"
            )
        self._model = model

    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> list[list[float]]:
        """Embed `texts`, returning L2-normalized vectors in input order.

        Normalizing here means cosine similarity is a dot product, matching the
        `cosine` similarity declared on the vector index.
        """
        if not texts:
            return []
        self.load()
        assert self._model is not None  # narrowed by load()
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size or get_settings().embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vector] for vector in vectors]
