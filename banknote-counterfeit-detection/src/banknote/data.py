"""Download, validate and split the UCI Banknote Authentication dataset."""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    COLUMNS,
    DATA_FILENAME,
    DEFAULT_DATA_DIR,
    FEATURES,
    MIRROR_URL,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    UCI_URL,
)


@dataclass(frozen=True)
class Split:
    """A stratified train/test split. The test set plays the role of new bills."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def download_banknotes(dest: Path, urls: tuple[str, ...] = (UCI_URL, MIRROR_URL), timeout: int = 30) -> Path:
    """Download the raw CSV to ``dest``, trying each URL in order."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (fixed https URLs)
                dest.write_bytes(response.read())
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the dataset:\n  " + "\n  ".join(errors))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail fast if the data does not look like the banknote dataset."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    labels = set(df[TARGET].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Target must be binary 0/1, got {sorted(labels)}")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES):
        raise ValueError("All feature columns must be numeric")
    return df


def load_banknotes(data_dir: Path | str = DEFAULT_DATA_DIR, download: bool = True) -> pd.DataFrame:
    """Load the dataset from ``data_dir``, downloading it on first use."""
    path = Path(data_dir) / DATA_FILENAME
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found and download=False")
        download_banknotes(path)
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return validate(df)


def make_split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Split:
    """Stratified split so both sets keep the genuine/counterfeit ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
    return Split(X_train, X_test, y_train, y_test)


def add_simulated_volume(
    X: pd.DataFrame, low: int = 10_000, high: int = 80_000, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Append a fake ``atm_volume`` column (daily transactions at the ATM).

    It carries no signal about the bill itself. It exists to show what happens
    to distance-based models when one feature is thousands of times larger
    than the others.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    out["atm_volume"] = rng.integers(low, high, size=len(X))
    return out
