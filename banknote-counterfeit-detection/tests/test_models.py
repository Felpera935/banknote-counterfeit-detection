import pytest

from banknote.evaluate import cross_validate_model, evaluate_on_test
from banknote.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_trains_and_beats_chance(split, name):
    model = build_model(name).fit(split.X_train, split.y_train)
    assert model.score(split.X_test, split.y_test) > 0.8


def test_tree_is_unscaled_by_default_and_others_are_scaled():
    assert "scaler" not in build_model("decision_tree").named_steps
    for name in set(MODEL_NAMES) - {"decision_tree"}:
        assert "scaler" in build_model(name).named_steps


def test_scale_flag_overrides_default():
    assert "scaler" not in build_model("knn", scale=False).named_steps
    assert "scaler" in build_model("decision_tree", scale=True).named_steps


def test_params_are_forwarded():
    assert build_model("decision_tree", max_depth=3).named_steps["model"].max_depth == 3
    assert build_model("knn", n_neighbors=7).named_steps["model"].n_neighbors == 7


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("random_forest")


def test_cross_validation_is_deterministic(split):
    a = cross_validate_model(build_model("decision_tree"), split.X_train, split.y_train, k=3)
    b = cross_validate_model(build_model("decision_tree"), split.X_train, split.y_train, k=3)
    assert a == b
    assert 0.0 <= a["cv_accuracy_mean"] <= 1.0


def test_confusion_counts_add_up(split):
    model = build_model("logistic_regression").fit(split.X_train, split.y_train)
    m = evaluate_on_test(model, split.X_test, split.y_test)
    total = m["genuine_accepted"] + m["genuine_rejected"] + m["fakes_accepted"] + m["fakes_caught"]
    assert total == len(split.y_test)
    assert m["fakes_accepted"] + m["fakes_caught"] == int(split.y_test.sum())
