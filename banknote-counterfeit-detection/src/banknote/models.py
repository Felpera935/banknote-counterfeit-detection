"""Model factory: every model is a scikit-learn Pipeline, so scaling is fit
inside each cross-validation fold and never leaks information from the
validation data."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

MODEL_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "knn",
    "linear_svm",
    "svm_rbf",
)

DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "knn": "k-NN (k=5)",
    "linear_svm": "Linear SVM",
    "svm_rbf": "SVM (RBF kernel)",
}

# Trees split on thresholds, so standardization does not change their
# predictions. Leaving the tree unscaled keeps its rules in the original
# sensor units, which is what an ATM firmware engineer needs.
_SCALE_BY_DEFAULT: dict[str, bool] = {name: name != "decision_tree" for name in MODEL_NAMES}


def _make_estimator(name: str, random_state: int, **params: Any) -> ClassifierMixin:
    if name == "logistic_regression":
        return LogisticRegression(random_state=random_state, max_iter=1000, **params)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    if name == "knn":
        return KNeighborsClassifier(**{"n_neighbors": 5, **params})
    if name == "linear_svm":
        return LinearSVC(random_state=random_state, max_iter=10_000, **params)
    if name == "svm_rbf":
        return SVC(random_state=random_state, **params)
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")


def build_model(
    name: str,
    *,
    scale: bool | None = None,
    random_state: int = RANDOM_STATE,
    **params: Any,
) -> Pipeline:
    """Return an unfitted pipeline ``[StandardScaler] -> classifier``.

    Args:
        name: one of ``MODEL_NAMES``.
        scale: force scaling on/off. ``None`` uses the sensible default
            (on for every model except the decision tree).
        random_state: seed for models with internal randomness.
        **params: forwarded to the classifier (e.g. ``max_depth=4``).
    """
    estimator = _make_estimator(name, random_state, **params)
    use_scaler = _SCALE_BY_DEFAULT[name] if scale is None else scale
    steps = [("scaler", StandardScaler())] if use_scaler else []
    steps.append(("model", estimator))
    return Pipeline(steps)
