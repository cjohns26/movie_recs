"""Tests for movie_recs.recsys.als."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from movie_recs.recsys.als import (
    IdMap,
    RecsysArtifact,
    build_matrix,
    item_item_neighbors,
    load_artifact,
    save_artifact,
    train_als,
    train_item_item,
)
from movie_recs.recsys.evaluate import _batch_ids_to_movie_ids
from movie_recs.recsys.metrics import mean_metric, ndcg_at_k
from movie_recs.recsys.split import compute_cutoff, temporal_split


def test_id_map_from_ids_is_sorted_and_contiguous() -> None:
    id_map = IdMap.from_ids(pd.Series([30, 10, 20, 10]))
    assert id_map.idx_to_id == [10, 20, 30]
    assert id_map.id_to_idx == {10: 0, 20: 1, 30: 2}
    assert len(id_map) == 3


def test_build_matrix_confidence_weighting_and_shape() -> None:
    train = pd.DataFrame(
        [
            (1, 10, 5.0, 0),  # liked
            (1, 20, 2.0, 1),  # not liked (below threshold) -> excluded
            (2, 10, 4.0, 2),  # liked
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    user_map = IdMap.from_ids(train["userId"])
    item_map = IdMap.from_ids(train["movieId"])

    matrix = build_matrix(train, user_map, item_map)

    assert matrix.shape == (2, 2)
    # user 1 (idx 0), item 10 (idx 0): confidence = 1 + 40*5.0
    assert matrix[0, 0] == 1.0 + 40.0 * 5.0
    # the sub-threshold rating never became a nonzero entry
    assert matrix[0, 1] == 0.0
    assert matrix.nnz == 2


def test_als_recommend_excludes_seen_items_and_has_correct_length() -> None:
    # 4 users x 5 items; user 0 has liked items 0 and 1.
    train = pd.DataFrame(
        [
            (0, 0, 5.0, 0),
            (0, 1, 5.0, 1),
            (1, 1, 4.0, 2),
            (1, 2, 4.0, 3),
            (2, 2, 4.0, 4),
            (2, 3, 4.0, 5),
            (3, 3, 4.0, 6),
            (3, 4, 4.0, 7),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    user_map = IdMap.from_ids(train["userId"])
    item_map = IdMap.from_ids(train["movieId"])
    matrix = build_matrix(train, user_map, item_map)

    model = train_als(matrix, n_factors=4, iterations=5)

    # Batch (array) calling convention, matching evaluate.py — the single
    # scalar-userid path has a separate, less consistent padding behavior.
    # N is clamped to the catalog size (5): asking `implicit` for more
    # items than exist pads with a bogus repeated real id (score 0.0, no
    # sentinel) rather than -1, matching evaluate.py's own clamp.
    user_idx = np.array([user_map.id_to_idx[0]])
    seen = {0, 1}
    n_recs = min(10, len(item_map))
    ids, _ = model.recommend(user_idx, matrix[user_idx], N=n_recs, filter_already_liked_items=True)
    recs = _batch_ids_to_movie_ids(ids[0], item_map, seen)

    assert len(recs) == 3  # 5 items total minus 2 already-liked
    assert 0 not in recs
    assert 1 not in recs


def test_item_item_neighbors_excludes_self() -> None:
    train = pd.DataFrame(
        [
            (0, 0, 5.0, 0),
            (0, 1, 5.0, 1),
            (1, 0, 4.0, 2),
            (1, 1, 4.0, 3),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    user_map = IdMap.from_ids(train["userId"])
    item_map = IdMap.from_ids(train["movieId"])
    matrix = build_matrix(train, user_map, item_map)

    model = train_item_item(matrix, k=5)
    neighbors = item_item_neighbors(model, item_map, k=5)

    for item_id, its_neighbors in neighbors.items():
        assert item_id not in {n_id for n_id, _ in its_neighbors}


def test_artifact_save_and_load_round_trips(tmp_path: Path) -> None:
    item_map = IdMap(id_to_idx={10: 0, 20: 1}, idx_to_id=[10, 20])
    artifact = RecsysArtifact(
        item_factors=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        regularization=0.05,
        n_factors=2,
        confidence_alpha=40.0,
        like_threshold=4.0,
        item_map=item_map,
        item_item_neighbors={10: [(20, 0.9)]},
        train_cutoff_ts=12345,
        n_train_interactions=2,
    )
    path = tmp_path / "artifacts" / "model.pkl"

    save_artifact(path, artifact)
    loaded = load_artifact(path)

    assert loaded.regularization == artifact.regularization
    assert loaded.item_map == artifact.item_map
    assert loaded.item_item_neighbors == artifact.item_item_neighbors
    np.testing.assert_array_equal(loaded.item_factors, artifact.item_factors)


def test_als_beats_popularity_on_ndcg10() -> None:
    """Guards against a broken pipeline: on data with real latent structure
    (two disjoint user/item clusters), a personalized model must beat a
    single global popularity ranking. Synthetic, not the real MovieLens
    fixture, because a robust ALS-vs-popularity gap needs a real signal
    for ALS to actually recover.
    """
    rng = np.random.default_rng(7)
    n_users_per_cluster = 20
    n_items_per_cluster = 20
    like_threshold = 4.0

    rows = []
    uid = 0
    for cluster in range(2):
        item_offset = cluster * n_items_per_cluster
        for _ in range(n_users_per_cluster):
            uid += 1
            items = rng.choice(n_items_per_cluster, size=14, replace=False) + item_offset
            for item in items:
                rating = min(4.0 + rng.random(), 5.0)
                rows.append((uid, int(item), float(rating), 0))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings.sample(frac=1.0, random_state=1).reset_index(drop=True)
    ratings["timestamp"] = range(len(ratings))

    cutoff = compute_cutoff(ratings, test_fraction=0.2)
    train, test = temporal_split(ratings, cutoff)
    train_liked = train[train["rating"] >= like_threshold]
    test_liked = test[test["rating"] >= like_threshold]

    user_map = IdMap.from_ids(train_liked["userId"])
    item_map = IdMap.from_ids(train_liked["movieId"])
    matrix = build_matrix(train, user_map, item_map)
    model = train_als(matrix, n_factors=8, iterations=20)

    relevant_by_user: dict[int, set[int]] = {}
    for raw_uid, group in test_liked.groupby("userId"):
        u = int(cast(int, raw_uid))
        if u not in user_map.id_to_idx:
            continue
        relevant = set(group["movieId"]) & set(item_map.id_to_idx)
        if relevant:
            relevant_by_user[u] = relevant
    assert len(relevant_by_user) > 10  # sanity: enough signal for the assertion to mean something

    pop_order = train_liked["movieId"].value_counts().index.tolist()
    seen_by_user = {int(cast(int, u)): set(g["movieId"]) for u, g in train_liked.groupby("userId")}

    eval_users = list(relevant_by_user)
    user_indices = np.array([user_map.id_to_idx[u] for u in eval_users])
    n_recs = min(10, len(item_map))
    ids, _ = model.recommend(
        user_indices, matrix[user_indices], N=n_recs, filter_already_liked_items=True
    )

    als_ndcgs = []
    pop_ndcgs = []
    for row, u in enumerate(eval_users):
        als_recs = _batch_ids_to_movie_ids(ids[row], item_map, seen_by_user.get(u))
        als_ndcgs.append(ndcg_at_k(als_recs, relevant_by_user[u], 10))

        pop_recs = [item for item in pop_order if item not in seen_by_user.get(u, set())][:10]
        pop_ndcgs.append(ndcg_at_k(pop_recs, relevant_by_user[u], 10))

    assert mean_metric(als_ndcgs) > mean_metric(pop_ndcgs) + 0.2  # comfortable margin
