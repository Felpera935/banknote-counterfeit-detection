import pandas as pd
import pytest

from banknote.config import COLUMNS, DATA_FILENAME, FEATURES, TARGET
from banknote.data import add_simulated_volume, load_banknotes, make_split, validate


def test_validate_accepts_good_data(synthetic_df):
    assert validate(synthetic_df) is synthetic_df


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda df: df.drop(columns=["entropy"]),
        lambda df: df.assign(variance=df["variance"].where(df.index != 3)),
        lambda df: df.assign(is_fake=df[TARGET] + 1),
    ],
    ids=["missing-column", "missing-value", "non-binary-target"],
)
def test_validate_rejects_bad_data(synthetic_df, corrupt):
    with pytest.raises(ValueError):
        validate(corrupt(synthetic_df))


def test_split_is_stratified_and_disjoint(synthetic_df):
    s = make_split(synthetic_df, test_size=0.2)
    assert len(s.X_test) == 60
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert list(s.X_train.columns) == FEATURES
    overall = synthetic_df[TARGET].mean()
    assert abs(s.y_train.mean() - overall) < 0.02
    assert abs(s.y_test.mean() - overall) < 0.02


def test_simulated_volume_is_reproducible_and_in_range(synthetic_df):
    X = synthetic_df[FEATURES]
    a = add_simulated_volume(X, seed=1)
    b = add_simulated_volume(X, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["atm_volume"].between(10_000, 80_000).all()
    assert "atm_volume" not in X.columns  # input left untouched


def test_load_reads_cached_file_without_download(tmp_path, synthetic_df):
    synthetic_df[COLUMNS].to_csv(tmp_path / DATA_FILENAME, header=False, index=False)
    df = load_banknotes(tmp_path, download=False)
    assert df.shape == synthetic_df.shape


def test_load_without_file_and_no_download_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_banknotes(tmp_path, download=False)
