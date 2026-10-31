"""Project-wide constants: schema, data sources and reproducibility settings."""

from __future__ import annotations

from pathlib import Path

# Four wavelet-transform statistics extracted from an image of each banknote.
FEATURES: list[str] = ["variance", "skewness", "kurtosis", "entropy"]

# 1 = counterfeit, 0 = genuine (the convention used throughout this project).
TARGET: str = "is_fake"
COLUMNS: list[str] = FEATURES + [TARGET]
CLASS_NAMES: dict[int, str] = {0: "genuine", 1: "counterfeit"}

DATA_FILENAME = "data_banknote_authentication.txt"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
# Public mirror of the same 1,372 rows, used only if the UCI server is unreachable.
MIRROR_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"

RANDOM_STATE = 0
TEST_SIZE = 0.2
CV_FOLDS = 5

# Business constraint: the ATM microchip can evaluate at most this many
# consecutive IF/THEN checks before the transaction times out.
ATM_MAX_DEPTH = 4

# Bills whose SVM decision value falls inside +/- this margin go to a human.
REVIEW_MARGIN = 1.0

DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_REPORTS_DIR = Path("reports")
