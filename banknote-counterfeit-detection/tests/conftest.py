"""Shared fixtures. Tests run on synthetic data, so they need no network access."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from banknote.config import FEATURES, TARGET
from banknote.data import make_split


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    df = pd.DataFrame(X * 4, columns=FEATURES)  # scale to the real data's order of magnitude
    df[TARGET] = y
    return df


@pytest.fixture
def split(synthetic_df):
    return make_split(synthetic_df)
