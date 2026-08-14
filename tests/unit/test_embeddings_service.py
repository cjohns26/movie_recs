"""Tests for the embedding microservice's HTTP contract.

The real encoder is replaced by a deterministic stub via FastAPI's
`dependency_overrides`, so these assert the contract (shape, ordering,
validation) without a GPU, a model download, or torch installed.
"""

from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from movie_recs.embeddings.service import MAX_BATCH, app, get_text_embedder
from movie_recs.embeddings.text import TEXT_EMBED_DIM, TextEmbedder


class StubEmbedder(TextEmbedder):
    """Encodes each text to a distinct constant vector keyed on its length."""

    def __init__(self) -> None:
        super().__init__(model_name="stub-model")
        self.calls: list[list[str]] = []

    def load(self) -> None:
        """No model to load — keeps startup warmup off the network."""

    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] * TEXT_EMBED_DIM for text in texts]


@pytest.fixture
def stub() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def client(stub: StubEmbedder) -> Iterator[TestClient]:
    app.dependency_overrides[get_text_embedder] = lambda: stub
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_embed_text_returns_a_384_dim_vector(client: TestClient) -> None:
    response = client.post("/embed/text", json={"texts": ["a movie about robots"]})

    assert response.status_code == 200
    body = response.json()
    assert body["dim"] == TEXT_EMBED_DIM
    assert body["model"] == "stub-model"
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == TEXT_EMBED_DIM


def test_batch_preserves_input_order(client: TestClient) -> None:
    """Callers zip the response against their own inputs (the catalog job pairs
    vectors back to movie ids positionally), so order is part of the contract."""
    texts = ["a", "bb", "ccc"]

    response = client.post("/embed/text", json={"texts": texts})

    embeddings = response.json()["embeddings"]
    assert [vector[0] for vector in embeddings] == [1.0, 2.0, 3.0]


def test_empty_batch_is_rejected(client: TestClient) -> None:
    response = client.post("/embed/text", json={"texts": []})

    assert response.status_code == 422


def test_oversized_batch_is_rejected(client: TestClient, stub: StubEmbedder) -> None:
    """A single request must not be able to pin the GPU with an unbounded batch."""
    response = client.post("/embed/text", json={"texts": ["x"] * (MAX_BATCH + 1)})

    assert response.status_code == 422
    assert stub.calls == []


def test_health_reports_the_model_and_dimension(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "text_model": "stub-model",
        "text_dim": TEXT_EMBED_DIM,
    }
