import json

from banknote import experiments as ex
from banknote import report
from banknote.config import FEATURES
from banknote.models import MODEL_NAMES


def test_compare_models_returns_one_sorted_row_per_model(split):
    results = ex.compare_models(split, k=3)
    assert set(results["model"]) == set(MODEL_NAMES)
    assert results["cv_accuracy_mean"].is_monotonic_decreasing


def test_logistic_explanation_covers_all_features(split):
    coefs = ex.explain_logistic(split)
    assert sorted(coefs["feature"]) == sorted(FEATURES)
    assert set(coefs["pushes_towards"]) <= {"genuine", "counterfeit"}


def test_depth_tradeoff_respects_limits(split):
    df = ex.depth_tradeoff(split, depths=[1, 2, 3], k=3)
    assert len(df) == 4  # three limited trees + the unconstrained one
    limited = df.dropna(subset=["max_depth"])
    assert (limited["actual_depth"] <= limited["max_depth"]).all()
    assert df["max_depth"].isna().iloc[-1]


def test_atm_rules_never_exceed_the_depth_limit(split):
    rules = ex.atm_rules(split, max_depth=2)
    deepest_check = max(line.count("|   ") for line in rules.splitlines() if "<=" in line or ">" in line)
    assert deepest_check <= 1  # depth 2 = root check + one nested check
    assert "class:" in rules


def test_scaling_experiment_has_all_four_combinations(split):
    df = ex.scaling_experiment(split, k=3)
    assert len(df) == 4
    assert set(df["scaling"]) == {"scaled", "unscaled"}


def test_borderline_bills_are_inside_the_margin(split):
    decisions = ex.svm_decision_values(split)
    review = ex.borderline_bills(decisions, margin=1.0)
    assert (review["decision_value"].abs() < 1.0).all()
    assert review["decision_value"].abs().is_monotonic_increasing
    outside = decisions["decision_value"].abs() >= 1.0
    assert len(review) == int((~outside).sum())


def test_run_review_writes_outputs(split, tmp_path):
    summary = report.run_review(split, tmp_path)
    assert (tmp_path / "borderline_bills.csv").exists()
    assert (tmp_path / "figures" / "svm_review_zone.png").exists()
    json.dumps(summary)  # summary must be JSON-serializable
