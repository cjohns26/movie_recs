"""Tests for movie_recs.recsys.split."""

import pandas as pd

from movie_recs.recsys.split import compute_cutoff, leave_one_out_split, temporal_split


def _ratings(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_compute_cutoff_is_a_quantile_of_timestamp() -> None:
    ratings = _ratings([(1, 1, 4.0, ts) for ts in range(100)])
    cutoff = compute_cutoff(ratings, test_fraction=0.2)
    assert cutoff == 79  # 0.8 quantile of 0..99


def test_temporal_split_has_no_leakage() -> None:
    """Every test interaction's timestamp is >= cutoff and > every train
    interaction's timestamp for the same user (the leakage guarantee)."""
    rows = [
        (1, 1, 5.0, 10),
        (1, 2, 4.0, 20),
        (1, 3, 3.0, 90),
        (2, 4, 4.5, 15),
        (2, 5, 5.0, 95),
    ]
    ratings = _ratings(rows)
    cutoff = compute_cutoff(ratings, test_fraction=0.4)

    train, test = temporal_split(ratings, cutoff)

    assert (train["timestamp"] < cutoff).all()
    assert (test["timestamp"] >= cutoff).all()

    train_max_by_user = train.groupby("userId")["timestamp"].max()
    for _, test_row in test.iterrows():
        assert test_row["timestamp"] >= cutoff
        user_train_max = train_max_by_user.get(test_row["userId"])
        if user_train_max is not None:
            assert test_row["timestamp"] > user_train_max


def test_temporal_split_partitions_all_rows_exactly_once() -> None:
    ratings = _ratings([(1, i, 4.0, i) for i in range(50)])
    cutoff = compute_cutoff(ratings)
    train, test = temporal_split(ratings, cutoff)
    assert len(train) + len(test) == len(ratings)


def test_leave_one_out_split_holds_out_each_users_most_recent_liked_item() -> None:
    rows = [
        (1, 10, 5.0, 100),  # user 1's most recent liked -> test
        (1, 11, 4.0, 50),
        (1, 12, 2.0, 200),  # newer than the liked ones, but below threshold -> stays train
        (2, 20, 4.5, 10),  # user 2's only liked interaction -> test
    ]
    ratings = _ratings(rows)

    train, test = leave_one_out_split(ratings, like_threshold=4.0)

    assert len(train) + len(test) == len(ratings)
    assert set(test["movieId"]) == {10, 20}
    assert set(train["movieId"]) == {11, 12}


def test_leave_one_out_split_covers_nearly_all_users_unlike_a_single_global_cutoff() -> None:
    """The whole point of the secondary split: every user with >=2 liked
    ratings is evaluable, regardless of how their activity clusters in
    time (unlike a single global cutoff on bursty per-user activity)."""
    rows = []
    for uid in range(1, 21):
        # each user's ratings all cluster in a narrow, user-specific window
        base = uid * 1000
        rows.append((uid, uid * 10, 5.0, base))
        rows.append((uid, uid * 10 + 1, 4.0, base + 1))

    ratings = _ratings(rows)
    _, test = leave_one_out_split(ratings, like_threshold=4.0)

    assert len(test) == 20  # every one of the 20 users is evaluable
