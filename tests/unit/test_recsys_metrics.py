"""Tests for movie_recs.recsys.metrics — values hand-computed independently
of the implementation (see the docstring formulas in metrics.py)."""

import math

import pytest

from movie_recs.recsys.metrics import (
    average_precision_at_k,
    catalog_coverage,
    intra_list_diversity,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = [10, 20, 30]
RELEVANT = {20, 30, 40}


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)


def test_recall_at_k_empty_relevant_is_zero_not_a_crash() -> None:
    assert recall_at_k(RANKED, set(), 2) == 0.0


def test_ndcg_at_k() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(4)  # hits at rank 2 (item 20), rank 3 (item 30)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant items, k=3
    expected = dcg / idcg
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(expected)


def test_ndcg_at_k_no_hits_is_zero() -> None:
    assert ndcg_at_k(RANKED, {999}, 3) == 0.0


def test_average_precision_at_k() -> None:
    expected = (1 / 2 + 2 / 3) / 3  # hits at rank 2 (precision 1/2), rank 3 (precision 2/3)
    assert average_precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(expected)


def test_mean_metric() -> None:
    assert mean_metric([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert mean_metric([]) == 0.0


def test_catalog_coverage() -> None:
    recs = [[1, 2, 3], [3, 4], [1]]
    assert catalog_coverage(recs, catalog_size=10) == pytest.approx(0.4)  # {1,2,3,4} / 10


def test_intra_list_diversity_known_set() -> None:
    item_genres = {
        1: {"Comedy", "Romance"},
        2: {"Comedy"},
        3: {"Horror"},
    }
    # pair(1,2): jaccard = |{"Comedy"}| / |{"Comedy","Romance"}| = 1/2 -> distance 0.5
    # pair(1,3): jaccard = 0 -> distance 1.0
    # pair(2,3): jaccard = 0 -> distance 1.0
    expected = (0.5 + 1.0 + 1.0) / 3
    assert intra_list_diversity([1, 2, 3], item_genres) == pytest.approx(expected)


def test_intra_list_diversity_single_item_is_zero() -> None:
    assert intra_list_diversity([1], {1: {"Comedy"}}) == 0.0
