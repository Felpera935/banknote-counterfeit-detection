"""Run experiments and persist their outputs (CSV tables, figures, rules, summary)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from . import experiments as ex
from . import plots
from .config import ATM_MAX_DEPTH, CV_FOLDS, RANDOM_STATE, REVIEW_MARGIN, TEST_SIZE
from .data import Split, load_banknotes, make_split


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(path, index=False)


def run_compare(split: Split, out_dir: Path, k: int = CV_FOLDS) -> dict[str, Any]:
    results = ex.compare_models(split, k=k)
    _write_csv(results, out_dir / "metrics" / "model_comparison.csv")
    plots.plot_model_comparison(results, out_dir / "figures" / "model_comparison.png")
    plots.plot_confusion_matrices(results, out_dir / "figures" / "confusion_matrices.png")
    best = results.iloc[0]
    return {
        "best_model_by_cv": best["model"],
        "models": {
            r["model"]: {
                "cv_accuracy": round(r["cv_accuracy_mean"], 4),
                "cv_accuracy_std": round(r["cv_accuracy_std"], 4),
                "cv_recall": round(r["cv_recall_mean"], 4),
                "test_accuracy": round(r["test_accuracy"], 4),
                "test_recall": round(r["test_recall"], 4),
                "fakes_accepted": int(r["fakes_accepted"]),
                "genuine_rejected": int(r["genuine_rejected"]),
            }
            for _, r in results.iterrows()
        },
    }


def run_explain(split: Split, out_dir: Path) -> dict[str, Any]:
    coefs = ex.explain_logistic(split)
    _write_csv(coefs, out_dir / "metrics" / "logistic_coefficients.csv")
    plots.plot_logistic_coefficients(coefs, out_dir / "figures" / "logistic_coefficients.png")
    return {"logistic_coefficients": dict(zip(coefs["feature"], coefs["coefficient"].round(3), strict=True))}


def run_depth(split: Split, out_dir: Path, k: int = CV_FOLDS, atm_limit: int = ATM_MAX_DEPTH) -> dict[str, Any]:
    depth_df = ex.depth_tradeoff(split, k=k)
    _write_csv(depth_df, out_dir / "metrics" / "depth_tradeoff.csv")
    plots.plot_depth_tradeoff(depth_df, atm_limit, out_dir / "figures" / "depth_tradeoff.png")

    at_limit = depth_df[depth_df["max_depth"] == atm_limit].iloc[0]
    unconstrained = depth_df[depth_df["max_depth"].isna()].iloc[0]
    rules = ex.atm_rules(split, max_depth=atm_limit)
    header = (
        f"# Decision tree limited to {atm_limit} consecutive checks (sensor units, not scaled)\n"
        f"# Hold-out accuracy: {at_limit['test_accuracy']:.4f} | "
        f"counterfeits accepted: {int(at_limit['fakes_accepted'])} | leaves: {int(at_limit['n_leaves'])}\n"
        "# Some splits lead to the same class on both sides: they make leaves purer\n"
        "# but do not change any prediction, so firmware can merge them.\n\n"
    )
    rules_path = out_dir / "atm_rules.txt"
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text(header + rules, encoding="utf-8")
    return {
        "atm_limited_tree": {
            "max_depth": atm_limit,
            "cv_accuracy": round(float(at_limit["cv_accuracy_mean"]), 4),
            "test_accuracy": round(float(at_limit["test_accuracy"]), 4),
            "fakes_accepted": int(at_limit["fakes_accepted"]),
        },
        "unconstrained_tree": {
            "depth": int(unconstrained["actual_depth"]),
            "cv_accuracy": round(float(unconstrained["cv_accuracy_mean"]), 4),
            "test_accuracy": round(float(unconstrained["test_accuracy"]), 4),
            "fakes_accepted": int(unconstrained["fakes_accepted"]),
        },
    }


def run_scaling(split: Split, out_dir: Path, k: int = CV_FOLDS) -> dict[str, Any]:
    scaling_df = ex.scaling_experiment(split, k=k)
    _write_csv(scaling_df, out_dir / "metrics" / "knn_scaling.csv")
    majority_rate = float((split.y_test == 0).mean())
    plots.plot_scaling(scaling_df, majority_rate, out_dir / "figures" / "knn_scaling.png")
    return {
        "knn_scaling": {
            f"{r['feature_set']} / {r['scaling']}": round(r["test_accuracy"], 4) for _, r in scaling_df.iterrows()
        },
        "majority_class_rate": round(majority_rate, 4),
    }


def run_review(split: Split, out_dir: Path, margin: float = REVIEW_MARGIN) -> dict[str, Any]:
    decisions = ex.svm_decision_values(split)
    review = ex.borderline_bills(decisions, margin)
    _write_csv(review, out_dir / "borderline_bills.csv")
    plots.plot_decision_values(decisions, margin, out_dir / "figures" / "svm_review_zone.png")
    return {
        "human_review": {
            "margin": margin,
            "bills_flagged": len(review),
            "bills_total": len(decisions),
            "share_flagged": round(len(review) / len(decisions), 4),
            "flagged_misclassified_by_linear_svm": int((~review["correct"]).sum()),
            "linear_svm_errors_outside_zone": int(
                (~decisions.loc[decisions["decision_value"].abs() >= margin, "correct"]).sum()
            ),
        }
    }


def run_all(
    data_dir: Path,
    out_dir: Path,
    k: int = CV_FOLDS,
    download: bool = True,
) -> dict[str, Any]:
    df = load_banknotes(data_dir, download=download)
    split = make_split(df)
    summary: dict[str, Any] = {
        "dataset": {
            "rows": len(df),
            "genuine": int((df["is_fake"] == 0).sum()),
            "counterfeit": int((df["is_fake"] == 1).sum()),
            "train_rows": len(split.y_train),
            "test_rows": len(split.y_test),
        },
        "settings": {
            "test_size": TEST_SIZE,
            "cv_folds": k,
            "random_state": RANDOM_STATE,
            "scikit_learn": sklearn.__version__,
        },
    }
    summary.update(run_compare(split, out_dir, k))
    summary.update(run_explain(split, out_dir))
    summary.update(run_depth(split, out_dir, k))
    summary.update(run_scaling(split, out_dir, k))
    summary.update(run_review(split, out_dir))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
