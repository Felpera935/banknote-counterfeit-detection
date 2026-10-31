"""Cross-validation and hold-out evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, RANDOM_STATE

# "Positive" = counterfeit, so recall answers: of all fake bills, how many did we catch?
SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def cv_splitter(k: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Shuffled, stratified folds: each fold keeps the class ratio."""
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Mean and standard deviation of every metric in ``SCORING`` over k folds."""
    results = cross_validate(clone(pipeline), X, y, cv=cv_splitter(k, random_state), scoring=SCORING, n_jobs=None)
    summary: dict[str, float] = {}
    for metric in SCORING:
        scores = results[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = float(np.mean(scores))
        summary[f"cv_{metric}_std"] = float(np.std(scores))
    return summary


def continuous_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Signed score for ROC AUC: decision function if available, else P(fake)."""
    if hasattr(pipeline, "decision_function"):
        try:
            return pipeline.decision_function(X)
        except AttributeError:
            pass
    return pipeline.predict_proba(X)[:, 1]


def evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Metrics of a *fitted* pipeline on the hold-out set, including the
    confusion matrix cells in business terms."""
    y_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, continuous_scores(pipeline, X_test))),
        "genuine_accepted": int(tn),
        "genuine_rejected": int(fp),  # false positive: customer's real bill refused
        "fakes_accepted": int(fn),  # false negative: counterfeit enters the ATM
        "fakes_caught": int(tp),
    }
