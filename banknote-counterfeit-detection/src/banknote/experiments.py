"""The four questions of the case study, each as a pure function that returns
data (no files, no plots). See ``report.py`` for persistence and figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from .config import ATM_MAX_DEPTH, CLASS_NAMES, CV_FOLDS, FEATURES, RANDOM_STATE, REVIEW_MARGIN
from .data import Split, add_simulated_volume
from .evaluate import cross_validate_model, evaluate_on_test
from .models import DISPLAY_NAMES, MODEL_NAMES, build_model


# 1. Which model should the ATM run? -------------------------------------------------
def compare_models(
    split: Split,
    models: tuple[str, ...] = MODEL_NAMES,
    k: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cross-validate every model on the training set, then refit on the full
    training set and score once on the hold-out test set."""
    rows = []
    for name in models:
        pipeline = build_model(name, random_state=random_state)
        row = {"model": name, "display_name": DISPLAY_NAMES[name]}
        row.update(cross_validate_model(pipeline, split.X_train, split.y_train, k, random_state))
        pipeline.fit(split.X_train, split.y_train)
        row.update(evaluate_on_test(pipeline, split.X_test, split.y_test))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False, ignore_index=True)


# 2. Can we explain the baseline? ----------------------------------------------------
def explain_logistic(split: Split, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Standardized logistic-regression coefficients.

    Because features are standardized, each coefficient is the change in the
    log-odds of *counterfeit* for a one-standard-deviation increase in that
    feature, so magnitudes are comparable across features.
    """
    pipeline = build_model("logistic_regression", random_state=random_state).fit(split.X_train, split.y_train)
    coefs = pipeline.named_steps["model"].coef_[0]
    df = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": coefs,
            "odds_ratio_per_sd": np.exp(coefs),
        }
    )
    df["pushes_towards"] = np.where(df["coefficient"] > 0, CLASS_NAMES[1], CLASS_NAMES[0])
    return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# 3. What does the ATM depth limit cost us? ------------------------------------------
def depth_tradeoff(
    split: Split,
    depths: range | list[int] = range(1, 11),
    k: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Accuracy of decision trees as the maximum depth grows.

    The last row (``max_depth`` = NaN) is the unconstrained tree.
    """
    rows = []
    for depth in [*depths, None]:
        pipeline = build_model("decision_tree", max_depth=depth, random_state=random_state)
        cv = cross_validate_model(pipeline, split.X_train, split.y_train, k, random_state)
        pipeline.fit(split.X_train, split.y_train)
        tree = pipeline.named_steps["model"]
        test = evaluate_on_test(pipeline, split.X_test, split.y_test)
        rows.append(
            {
                "max_depth": depth,
                "actual_depth": tree.get_depth(),
                "n_leaves": tree.get_n_leaves(),
                "cv_accuracy_mean": cv["cv_accuracy_mean"],
                "cv_accuracy_std": cv["cv_accuracy_std"],
                "test_accuracy": test["test_accuracy"],
                "test_recall": test["test_recall"],
                "fakes_accepted": test["fakes_accepted"],
                "genuine_rejected": test["genuine_rejected"],
            }
        )
    df = pd.DataFrame(rows)
    df["max_depth"] = df["max_depth"].astype("Int64")
    return df


def atm_rules(split: Split, max_depth: int = ATM_MAX_DEPTH, random_state: int = RANDOM_STATE) -> str:
    """The depth-limited tree as plain IF/THEN rules in original sensor units."""
    pipeline = build_model("decision_tree", max_depth=max_depth, random_state=random_state).fit(
        split.X_train, split.y_train
    )
    tree = pipeline.named_steps["model"]
    return export_text(
        tree,
        feature_names=FEATURES,
        class_names=[CLASS_NAMES[c] for c in tree.classes_],
        decimals=3,
    )


# 4a. Why does scaling matter? -------------------------------------------------------
def scaling_experiment(
    split: Split,
    n_neighbors: int = 5,
    k: int = CV_FOLDS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """k-NN with and without standardization, before and after adding a large,
    irrelevant ``atm_volume`` feature (10,000 to 80,000 transactions/day)."""
    feature_sets = {
        "4 sensor features": (split.X_train, split.X_test),
        "+ ATM volume": (
            add_simulated_volume(split.X_train, seed=seed),
            add_simulated_volume(split.X_test, seed=seed + 1),
        ),
    }
    rows = []
    for feature_set, (X_train, X_test) in feature_sets.items():
        for scaled in (False, True):
            pipeline = build_model("knn", scale=scaled, n_neighbors=n_neighbors)
            cv = cross_validate_model(pipeline, X_train, split.y_train, k, seed)
            pipeline.fit(X_train, split.y_train)
            test = evaluate_on_test(pipeline, X_test, split.y_test)
            rows.append(
                {
                    "feature_set": feature_set,
                    "scaling": "scaled" if scaled else "unscaled",
                    "cv_accuracy_mean": cv["cv_accuracy_mean"],
                    "test_accuracy": test["test_accuracy"],
                    "fakes_accepted": test["fakes_accepted"],
                }
            )
    return pd.DataFrame(rows)


# 4b. Which bills should a human double-check? ---------------------------------------
def svm_decision_values(split: Split, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Signed distance of every test bill from a linear SVM boundary.

    Positive = counterfeit side. |value| < 1 means the bill sits inside the
    SVM margin, where the model is least certain.
    """
    pipeline = build_model("linear_svm", random_state=random_state).fit(split.X_train, split.y_train)
    out = split.X_test.copy()
    out["decision_value"] = pipeline.decision_function(split.X_test)
    out["predicted"] = pipeline.predict(split.X_test)
    out["actual"] = split.y_test.to_numpy()
    out["correct"] = out["predicted"] == out["actual"]
    return out


def borderline_bills(decisions: pd.DataFrame, margin: float = REVIEW_MARGIN) -> pd.DataFrame:
    """Bills inside the review margin, closest to the boundary first."""
    inside = decisions[decisions["decision_value"].abs() < margin].copy()
    inside["predicted"] = inside["predicted"].map(CLASS_NAMES)
    inside["actual"] = inside["actual"].map(CLASS_NAMES)
    order = inside["decision_value"].abs().sort_values().index
    return inside.loc[order].rename_axis("bill_id").reset_index()
