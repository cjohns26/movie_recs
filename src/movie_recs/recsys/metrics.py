"""Offline recommendation quality metrics.

Rank-unaware (Precision, Recall) and rank-aware (NDCG, MAP) are reported
together per plan.md's Evaluation Plan, plus catalog coverage and
intra-list diversity to guard against popularity collapse.
"""

import itertools
import math
from collections.abc import Sequence


def precision_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of all relevant items captured in the top-k recommendations."""
    if not relevant:
        return 0.0
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    top_k = ranked_items[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Average Precision at k: precision computed at each hit position
    within the top-k, averaged over min(|relevant|, k)."""
    if not relevant:
        return 0.0
    top_k = ranked_items[:k]
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    denom = min(len(relevant), k)
    return precision_sum / denom if denom > 0 else 0.0


def mean_metric(per_user_scores: Sequence[float]) -> float:
    """Mean of a per-user metric across all evaluated users."""
    return sum(per_user_scores) / len(per_user_scores) if per_user_scores else 0.0


def catalog_coverage(all_recommended: Sequence[Sequence[int]], catalog_size: int) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    if catalog_size == 0:
        return 0.0
    recommended_items: set[int] = set()
    for items in all_recommended:
        recommended_items.update(items)
    return len(recommended_items) / catalog_size


def intra_list_diversity(ranked_items: Sequence[int], item_genres: dict[int, set[str]]) -> float:
    """Mean pairwise genre-Jaccard *distance* between items in one list.

    Higher = more genre-diverse recommendations (guards against popularity
    collapse into a single genre). Items missing genre data are skipped.
    """
    items = [i for i in ranked_items if i in item_genres]
    if len(items) < 2:
        return 0.0
    distances = []
    for a, b in itertools.combinations(items, 2):
        genres_a, genres_b = item_genres[a], item_genres[b]
        union = genres_a | genres_b
        similarity = len(genres_a & genres_b) / len(union) if union else 0.0
        distances.append(1.0 - similarity)
    return sum(distances) / len(distances)
