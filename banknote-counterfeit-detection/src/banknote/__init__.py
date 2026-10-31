"""Counterfeit banknote detection with scikit-learn pipelines."""

from .data import Split, load_banknotes, make_split
from .models import MODEL_NAMES, build_model

__all__ = ["MODEL_NAMES", "Split", "build_model", "load_banknotes", "make_split"]
__version__ = "1.0.0"
