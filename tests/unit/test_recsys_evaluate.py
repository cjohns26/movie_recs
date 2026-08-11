"""Tests for movie_recs.recsys.evaluate's pure-logic helpers.

`run()` itself (Mongo I/O + training + printing) isn't unit tested here —
it's glue/orchestration exercised via the actual `evaluate` CLI run against
real data (see plan.md's Session 3 Demo), the same treatment `ingest.run`
got in Session 2.
"""

import numpy as np
import pandas as pd

from movie_recs.recsys.als import IdMap
from movie_recs.recsys.evaluate import (
    MetricRow,
    _batch_ids_to_movie_ids,
    evaluate_model,
    format_table,
    popularity_ranking,
    recommend_random,
    recommend_ranked,
    user_seen_map,
)


def test_popularity_ranking_orders_by_like_count_descending() -> None:
    train_liked = pd.DataFrame(
        {"userId": [1, 2, 3, 4], "movieId": [10, 10, 10, 20]},
    )
    assert popularity_ranking(train_liked) == [10, 20]


def test_user_seen_map_groups_by_user() -> None:
    train_liked = pd.DataFrame(
        {"userId": [1, 1, 2], "movieId": [10, 20, 30]},
    )
    seen = user_seen_map(train_liked)
    assert seen == {1: {10, 20}, 2: {30}}


def test_recommend_ranked_skips_seen_items() -> None:
    recs = recommend_ranked([10, 20, 30, 40], seen={20}, n=2)
    assert recs == [10, 30]


def test_recommend_random_excludes_seen_and_respects_n() -> None:
    rng = np.random.default_rng(0)
    recs = recommend_random([1, 2, 3, 4, 5], seen={2, 4}, n=2, rng=rng)
    assert len(recs) == 2
    assert set(recs).isdisjoint({2, 4})
    assert set(recs).issubset({1, 3, 5})


def test_batch_ids_to_movie_ids_drops_sentinel_negative_one() -> None:
    item_map = IdMap(id_to_idx={10: 0, 20: 1}, idx_to_id=[10, 20])
    ids_row = np.array([1, 0, -1], dtype=np.int32)
    assert _batch_ids_to_movie_ids(ids_row, item_map) == [20, 10]


def test_batch_ids_to_movie_ids_defensively_drops_already_seen() -> None:
    """`implicit`'s own filter_already_liked_items was empirically found
    to leak already-liked items through on small/sparse matrices — this
    is the belt-and-suspenders re-filter for that."""
    item_map = IdMap(id_to_idx={10: 0, 20: 1, 30: 2}, idx_to_id=[10, 20, 30])
    ids_row = np.array([0, 1, 2], dtype=np.int32)  # movieIds 10, 20, 30
    assert _batch_ids_to_movie_ids(ids_row, item_map, seen={20}) == [10, 30]


def test_evaluate_model_aggregates_known_values() -> None:
    recs_by_user = {1: [10, 20, 30], 2: [20, 30, 10]}
    relevant_by_user = {1: {20}, 2: {20, 30}}
    item_genres = {10: {"Comedy"}, 20: {"Comedy"}, 30: {"Drama"}}

    row = evaluate_model("test_model", recs_by_user, relevant_by_user, item_genres, catalog_size=5)

    assert row["model"] == "test_model"
    # user1: P@10 = 1/10 (1 hit / k=10); user2: P@10 = 2/10 -> mean 0.15
    assert row["P@10"] == (0.1 + 0.2) / 2
    assert row["Coverage@10"] == 3 / 5  # {10,20,30} recommended out of 5 catalog items


def test_format_table_includes_header_and_all_rows() -> None:
    rows: list[MetricRow] = [
        {"model": "a", "P@10": 0.5},
        {"model": "b", "P@10": 0.25},
    ]
    table = format_table(rows)
    lines = table.splitlines()
    assert lines[0].startswith("model")
    assert len(lines) == 4  # header + separator + 2 rows
    assert "a" in lines[2]
    assert "0.2500" in lines[3]
