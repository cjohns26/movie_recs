"""Trains ALS, item-item, popularity, and random models on the real
ingested ratings, evaluates all four on both the primary (global temporal)
and secondary (per-user leave-one-out) splits, prints a metrics table for
each, and persists the primary split's ALS + item-item artifact for reuse.

Usage: uv run python -m movie_recs.recsys.evaluate [--artifact-path PATH]
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from numpy.typing import NDArray

from movie_recs.db.collections import movies_collection, ratings_collection
from movie_recs.recsys.als import (
    CONFIDENCE_ALPHA,
    LIKE_THRESHOLD,
    IdMap,
    RecsysArtifact,
    build_matrix,
    item_item_neighbors,
    save_artifact,
    train_als,
    train_item_item,
)
from movie_recs.recsys.metrics import (
    average_precision_at_k,
    catalog_coverage,
    intra_list_diversity,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from movie_recs.recsys.split import compute_cutoff, leave_one_out_split, temporal_split

logger = logging.getLogger(__name__)

K_VALUES = (10, 20)
TOP_N = max(K_VALUES)
RANDOM_STATE = 42
DEFAULT_ARTIFACT_PATH = Path("artifacts/recsys/model.pkl")

MetricRow = dict[str, float | str]


def load_ratings() -> pd.DataFrame:
    """Pull the full `ratings` collection into a DataFrame."""
    docs = list(
        ratings_collection().find(
            {}, {"_id": 0, "userId": 1, "movieId": 1, "rating": 1, "timestamp": 1}
        )
    )
    return pd.DataFrame(docs)


def load_item_genres() -> dict[int, set[str]]:
    """movieId -> set(genres), for the intra-list-diversity metric."""
    docs = movies_collection().find({}, {"_id": 1, "genres": 1})
    return {doc["_id"]: set(doc.get("genres", [])) for doc in docs}


def popularity_ranking(train_liked: pd.DataFrame) -> list[int]:
    """Movies ranked by train-set popularity (count of likes), descending."""
    return [int(i) for i in train_liked["movieId"].value_counts().index.tolist()]


def user_seen_map(train_liked: pd.DataFrame) -> dict[int, set[int]]:
    # groupby's key type is a broad Hashable union per pandas-stubs; "userId"
    # is always numeric at runtime.
    return {
        int(cast(int, uid)): set(group["movieId"]) for uid, group in train_liked.groupby("userId")
    }


def recommend_ranked(ranked_catalog: list[int], seen: set[int], n: int) -> list[int]:
    """Take the top-n items from a fixed ranking, skipping already-seen ones."""
    return [item for item in ranked_catalog if item not in seen][:n]


def recommend_random(
    catalog: list[int], seen: set[int], n: int, rng: np.random.Generator
) -> list[int]:
    candidates = [item for item in catalog if item not in seen]
    rng.shuffle(candidates)
    return candidates[:n]


def _batch_ids_to_movie_ids(
    ids_row: NDArray[np.int32], item_map: IdMap, seen: set[int] | None = None
) -> list[int]:
    """Map a `.recommend()` batch row's matrix indices back to movieIds.

    Drops two things `implicit` can hand back, neither safe to pass
    straight to `idx_to_id`:
    - the sentinel id -1, padded in when N exceeds the item catalog size
      (a bare -1 there would wrap to the *last* item);
    - already-`seen` items, defensively re-filtered here — on small/sparse
      matrices, `filter_already_liked_items=True` was empirically found to
      leak already-liked items through with a plain 0.0 score (no
      sentinel at all), so it can't be trusted alone.
    """
    seen = seen or set()
    return [
        item_map.idx_to_id[int(idx)]
        for idx in ids_row
        if idx != -1 and item_map.idx_to_id[int(idx)] not in seen
    ]


def evaluate_model(
    name: str,
    recs_by_user: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    item_genres: dict[int, set[str]],
    catalog_size: int,
) -> MetricRow:
    row: MetricRow = {"model": name}
    for k in K_VALUES:
        row[f"P@{k}"] = mean_metric(
            [precision_at_k(recs_by_user[u], relevant_by_user[u], k) for u in recs_by_user]
        )
        row[f"R@{k}"] = mean_metric(
            [recall_at_k(recs_by_user[u], relevant_by_user[u], k) for u in recs_by_user]
        )
        row[f"NDCG@{k}"] = mean_metric(
            [ndcg_at_k(recs_by_user[u], relevant_by_user[u], k) for u in recs_by_user]
        )
        row[f"MAP@{k}"] = mean_metric(
            [average_precision_at_k(recs_by_user[u], relevant_by_user[u], k) for u in recs_by_user]
        )
    row["Coverage@10"] = catalog_coverage(
        [recs[:10] for recs in recs_by_user.values()], catalog_size
    )
    row["Diversity@10"] = mean_metric(
        [intra_list_diversity(recs[:10], item_genres) for recs in recs_by_user.values()]
    )
    return row


def format_table(rows: list[MetricRow]) -> str:
    columns = list(rows[0].keys())

    def fmt_cell(value: float | str) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    widths = {c: max(len(c), *(len(fmt_cell(r[c])) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = [" | ".join(fmt_cell(r[c]).ljust(widths[c]) for c in columns) for r in rows]
    return "\n".join([header, sep, *body])


@dataclass
class SplitEvaluation:
    rows: list[MetricRow]
    als_model: AlternatingLeastSquares
    item_item_model: CosineRecommender
    item_map: IdMap
    n_train_interactions: int


def evaluate_split(
    train: pd.DataFrame, test: pd.DataFrame, item_genres: dict[int, set[str]]
) -> SplitEvaluation:
    """Train all four models on `train` and evaluate them against `test`.

    Shared by both the primary (global temporal) and secondary (per-user
    leave-one-out) splits — see plan.md's Evaluation Plan.
    """
    train_liked = train.loc[train["rating"] >= LIKE_THRESHOLD]
    test_liked = test.loc[test["rating"] >= LIKE_THRESHOLD]

    user_map = IdMap.from_ids(train_liked["userId"])
    item_map = IdMap.from_ids(train_liked["movieId"])
    catalog = item_map.idx_to_id
    catalog_set = set(item_map.id_to_idx)

    user_item = build_matrix(train, user_map, item_map)

    relevant_by_user: dict[int, set[int]] = {}
    for raw_uid, group in test_liked.groupby("userId"):
        # groupby's key type is a broad Hashable union per pandas-stubs;
        # "userId" is always numeric at runtime.
        uid = cast(int, raw_uid)
        if uid not in user_map.id_to_idx:
            continue  # no train signal for this user -> not CF-evaluable
        relevant = set(group["movieId"]) & catalog_set
        if relevant:
            relevant_by_user[int(uid)] = relevant

    eval_users = list(relevant_by_user)
    if not eval_users:
        raise RuntimeError(
            "No evaluable users after the split (need users with both train and test interactions)."
        )
    logger.info("Evaluating on %d users", len(eval_users))

    seen_by_user = user_seen_map(train_liked)

    als_model = train_als(user_item)
    item_item_model = train_item_item(user_item)

    user_indices = np.array([user_map.id_to_idx[u] for u in eval_users])
    rows_matrix = user_item[user_indices]
    # `implicit` pads with a bogus repeated real id (score 0.0, no
    # sentinel) rather than -1 when N exceeds the catalog size — never
    # ask for more than exists.
    n_recs = min(TOP_N, len(item_map))

    als_ids, _ = als_model.recommend(
        user_indices, rows_matrix, N=n_recs, filter_already_liked_items=True
    )
    ii_ids, _ = item_item_model.recommend(
        user_indices, rows_matrix, N=n_recs, filter_already_liked_items=True
    )

    als_recs = {
        uid: _batch_ids_to_movie_ids(als_ids[row], item_map, seen_by_user.get(uid))
        for row, uid in enumerate(eval_users)
    }
    ii_recs = {
        uid: _batch_ids_to_movie_ids(ii_ids[row], item_map, seen_by_user.get(uid))
        for row, uid in enumerate(eval_users)
    }

    pop_order = popularity_ranking(train_liked)
    pop_recs = {
        uid: recommend_ranked(pop_order, seen_by_user.get(uid, set()), TOP_N) for uid in eval_users
    }

    rng = np.random.default_rng(RANDOM_STATE)
    random_recs = {
        uid: recommend_random(catalog, seen_by_user.get(uid, set()), TOP_N, rng)
        for uid in eval_users
    }

    rows = [
        evaluate_model("popularity", pop_recs, relevant_by_user, item_genres, len(catalog)),
        evaluate_model("random", random_recs, relevant_by_user, item_genres, len(catalog)),
        evaluate_model("item_item", ii_recs, relevant_by_user, item_genres, len(catalog)),
        evaluate_model("als", als_recs, relevant_by_user, item_genres, len(catalog)),
    ]

    return SplitEvaluation(
        rows=rows,
        als_model=als_model,
        item_item_model=item_item_model,
        item_map=item_map,
        n_train_interactions=len(train_liked),
    )


def run(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> list[MetricRow]:
    ratings = load_ratings()
    item_genres = load_item_genres()

    cutoff = compute_cutoff(ratings)
    global_train, global_test = temporal_split(ratings, cutoff)
    global_eval = evaluate_split(global_train, global_test, item_genres)

    print("Primary split — global temporal cutoff (leakage-safe; see plan.md):")
    print(format_table(global_eval.rows))

    loo_train, loo_test = leave_one_out_split(ratings, LIKE_THRESHOLD)
    loo_eval = evaluate_split(loo_train, loo_test, item_genres)

    print()
    print("Secondary split — per-user leave-one-out (more evaluable users; see plan.md):")
    print(format_table(loo_eval.rows))

    neighbors = item_item_neighbors(global_eval.item_item_model, global_eval.item_map)
    artifact = RecsysArtifact(
        item_factors=global_eval.als_model.item_factors,
        regularization=global_eval.als_model.regularization,
        n_factors=global_eval.als_model.factors,
        confidence_alpha=CONFIDENCE_ALPHA,
        like_threshold=LIKE_THRESHOLD,
        item_map=global_eval.item_map,
        item_item_neighbors=neighbors,
        train_cutoff_ts=cutoff,
        n_train_interactions=global_eval.n_train_interactions,
    )
    save_artifact(artifact_path, artifact)
    logger.info("Artifact saved to %s (from the primary split)", artifact_path)

    return global_eval.rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train + evaluate CF models, print a metrics table"
    )
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()
    run(args.artifact_path)


if __name__ == "__main__":
    main()
