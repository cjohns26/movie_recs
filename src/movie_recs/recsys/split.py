"""Train/test splitting.

`temporal_split` (a single global timestamp cutoff) is the primary,
leakage-safe split — see plan.md's Evaluation Plan and Research Notes.
`leave_one_out_split` is the secondary, per-user view the same section
calls for; see its docstring for why it's secondary, not primary.
"""

import pandas as pd


def compute_cutoff(ratings: pd.DataFrame, test_fraction: float = 0.2) -> int:
    """The `test_fraction` quantile of `timestamp` across all ratings.

    Interactions at-or-after this cutoff are held out for evaluation;
    everything strictly before it is training data.
    """
    return int(ratings["timestamp"].quantile(1 - test_fraction))


def temporal_split(ratings: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by a single global timestamp cutoff.

    Every train interaction has `timestamp < cutoff`; every test
    interaction has `timestamp >= cutoff`. Since this holds per-row
    regardless of user, no test interaction can precede any train
    interaction for the same user — the no-leakage guarantee.
    """
    train = ratings.loc[ratings["timestamp"] < cutoff].reset_index(drop=True)
    test = ratings.loc[ratings["timestamp"] >= cutoff].reset_index(drop=True)
    return train, test


def leave_one_out_split(
    ratings: pd.DataFrame, like_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user leave-one-out: each user's single most recent "liked"
    (rating >= like_threshold) interaction becomes their test example;
    everything else is train.

    The secondary view plan.md's Evaluation Plan calls for alongside the
    primary global split. It's far more statistically robust — every user
    with >=2 liked ratings is evaluable, vs. the global split's small
    train/test user overlap on real, bursty MovieLens data (most users
    rate in a single time window, so a single global cutoff rarely falls
    inside any individual user's history) — but it's reported secondarily,
    not primary, because it can leak: a user's held-out item may still be
    timestamp-earlier than other users' training interactions, so a model
    can implicitly learn from data that's "in the future" relative to the
    item being predicted, in a way the global split structurally cannot.
    """
    liked = ratings.loc[ratings["rating"] >= like_threshold]
    last_idx = liked.groupby("userId")["timestamp"].idxmax().tolist()
    test = ratings.loc[last_idx].reset_index(drop=True)
    train = ratings.drop(index=last_idx).reset_index(drop=True)
    return train, test
