"""ALS matrix factorization + item-item cosine CF.

Both are trained on the same confidence-weighted implicit-feedback matrix:
ratings binarized at `LIKE_THRESHOLD` (matches `ingest.run`'s binarization),
with confidence `1 + CONFIDENCE_ALPHA * rating` for liked interactions
(Hu et al. 2008's implicit-feedback confidence formula). Unobserved entries
are implicit negatives, per `implicit`'s convention.
"""

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import threadpoolctl
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import coo_matrix, csr_matrix

LIKE_THRESHOLD = 4.0
CONFIDENCE_ALPHA = 40.0  # Hu et al. 2008's default confidence scaling
N_FACTORS = 64
REGULARIZATION = 0.05
ITERATIONS = 15
ITEM_NEIGHBORS_K = 20
RANDOM_STATE = 42


@dataclass
class IdMap:
    """Bidirectional MovieLens id <-> contiguous matrix-index mapping."""

    id_to_idx: dict[int, int]
    idx_to_id: list[int]

    @classmethod
    def from_ids(cls, ids: pd.Series) -> "IdMap":
        unique_ids = sorted(int(i) for i in ids.unique())
        return cls(id_to_idx={i: idx for idx, i in enumerate(unique_ids)}, idx_to_id=unique_ids)

    def __len__(self) -> int:
        return len(self.idx_to_id)


def build_matrix(train: pd.DataFrame, user_map: IdMap, item_map: IdMap) -> csr_matrix:
    """Build the confidence-weighted implicit-feedback user-item matrix.

    Only "liked" interactions (rating >= LIKE_THRESHOLD) become nonzero
    entries; confidence = 1 + CONFIDENCE_ALPHA * rating.
    """
    liked = train.loc[train["rating"] >= LIKE_THRESHOLD]
    rows = liked["userId"].map(user_map.id_to_idx).to_numpy()
    cols = liked["movieId"].map(item_map.id_to_idx).to_numpy()
    data = (1.0 + CONFIDENCE_ALPHA * liked["rating"]).to_numpy(dtype=np.float32)
    shape = (len(user_map), len(item_map))
    return coo_matrix((data, (rows, cols)), shape=shape).tocsr()


def train_als(
    user_item: csr_matrix,
    *,
    n_factors: int = N_FACTORS,
    regularization: float = REGULARIZATION,
    iterations: int = ITERATIONS,
) -> AlternatingLeastSquares:
    """Train ALS. `alpha` is intentionally left at the model's default
    (1.0) — confidence is already pre-scaled by CONFIDENCE_ALPHA in
    `build_matrix`, and the model's own `alpha` is just an additional
    multiplier on whatever matrix it's given, not the paper's `1 + a*r`
    formula (verified against `implicit.cpu.als` source).
    """
    model = AlternatingLeastSquares(
        factors=n_factors,
        regularization=regularization,
        iterations=iterations,
        random_state=RANDOM_STATE,
        num_threads=1,  # deterministic given a fixed random_state
    )
    with threadpoolctl.threadpool_limits(1, "blas"):
        model.fit(user_item, show_progress=False)
    return model


def train_item_item(user_item: csr_matrix, *, k: int = ITEM_NEIGHBORS_K) -> CosineRecommender:
    model = CosineRecommender(K=k, num_threads=1)
    with threadpoolctl.threadpool_limits(1, "blas"):
        model.fit(user_item, show_progress=False)
    return model


def item_item_neighbors(
    model: CosineRecommender, item_map: IdMap, *, k: int = ITEM_NEIGHBORS_K
) -> dict[int, list[tuple[int, float]]]:
    """Precompute top-k neighbors per item, keyed by real movieId.

    `similar_items` includes the query item itself (similarity 1.0) in its
    results — excluded here since it's not a useful "neighbor".
    """
    neighbors: dict[int, list[tuple[int, float]]] = {}
    for idx, item_id in enumerate(item_map.idx_to_id):
        ids, scores = model.similar_items(idx, N=k + 1)
        neighbors[item_id] = [
            (item_map.idx_to_id[int(n_idx)], float(score))
            for n_idx, score in zip(ids, scores, strict=True)
            if int(n_idx) != idx
        ][:k]
    return neighbors


@dataclass
class RecsysArtifact:
    """Everything needed to reuse the trained CF model without retraining.

    `item_factors` + `regularization` are reused verbatim by Session 7's
    ALS fold-in to solve a new session's user-vector against these fixed
    item factors.
    """

    item_factors: Any  # np.ndarray[float32], shape (n_items, n_factors)
    regularization: float
    n_factors: int
    confidence_alpha: float
    like_threshold: float
    item_map: IdMap
    item_item_neighbors: dict[int, list[tuple[int, float]]]
    train_cutoff_ts: int
    n_train_interactions: int
    trained_at: datetime = field(default_factory=datetime.now)


def save_artifact(path: Path, artifact: RecsysArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(artifact, f)


def load_artifact(path: Path) -> RecsysArtifact:
    with path.open("rb") as f:
        result: RecsysArtifact = pickle.load(f)
    return result
