"""
tests/test_pipeline.py
Basic test suite covering the core pipeline: preprocessing, model building,
prediction, and explanation. Run with:

    pytest tests/ -v

Note: test_prediction_and_explanation tests require models/ to already be
trained (they're skipped automatically if the trained model files aren't
present, e.g. in a fresh CI checkout before `python src/train.py` runs).
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import map_attack_category, COLUMN_NAMES, CATEGORY_LIST
from src.ensemble_model import build_stacking_classifier
from src.preprocess import load_raw, build_features
from src.utils import DATA_DIR, MODELS_DIR


# --------------------------------------------------------------------------
# utils.py
# --------------------------------------------------------------------------
def test_map_attack_category_known_labels():
    assert map_attack_category("normal") == "Normal"
    assert map_attack_category("neptune") == "DoS"
    assert map_attack_category("satan") == "Probe"
    assert map_attack_category("guess_passwd") == "R2L"
    assert map_attack_category("rootkit") == "U2R"


def test_map_attack_category_unseen_label_does_not_crash():
    assert map_attack_category("totally_new_attack_type") == "Unknown"


def test_map_attack_category_handles_trailing_dot_and_case():
    # NSL-KDD's original KDD Cup format sometimes has "normal." with a
    # trailing period; make sure that's handled gracefully.
    assert map_attack_category("Normal.") == "Normal"


def test_column_names_length():
    # 41 features + label + difficulty = 43 columns
    assert len(COLUMN_NAMES) == 43


def test_category_list_has_five_entries():
    assert len(CATEGORY_LIST) == 5
    assert "Normal" in CATEGORY_LIST


# --------------------------------------------------------------------------
# ensemble_model.py
# --------------------------------------------------------------------------
def test_build_stacking_classifier_has_three_base_learners():
    clf = build_stacking_classifier()
    names = [name for name, _ in clf.estimators]
    assert set(names) == {"random_forest", "xgboost", "mlp"}


def test_stacking_classifier_trains_and_predicts_on_toy_data():
    """Fast smoke test: does the ensemble actually fit and predict on tiny,
    synthetic, clearly-separable data, without needing the real dataset?"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] > 0).astype(int)  # trivially separable

    clf = build_stacking_classifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.85  # should easily solve this trivial task


# --------------------------------------------------------------------------
# preprocess.py (requires the raw NSL-KDD files to be present)
# --------------------------------------------------------------------------
DATA_PRESENT = os.path.exists(os.path.join(DATA_DIR, "KDDTrain+.txt")) and \
    os.path.exists(os.path.join(DATA_DIR, "KDDTest+.txt"))


@pytest.mark.skipif(not DATA_PRESENT, reason="run data/download_data.py first")
def test_build_features_shapes_match():
    train_df = load_raw(os.path.join(DATA_DIR, "KDDTrain+.txt"))
    test_df = load_raw(os.path.join(DATA_DIR, "KDDTest+.txt"))
    # use small slices for speed
    data = build_features(train_df.head(500), test_df.head(200))

    assert data["X_train"].shape[0] == 500
    assert data["X_test"].shape[0] == 200
    assert data["X_train"].shape[1] == data["X_test"].shape[1]
    assert len(data["y_train_bin"]) == 500
    assert set(np.unique(data["y_train_bin"])).issubset({0, 1})


# --------------------------------------------------------------------------
# predict.py / explain.py (require trained models)
# --------------------------------------------------------------------------
MODELS_PRESENT = os.path.exists(os.path.join(MODELS_DIR, "binary_model.pkl")) and \
    os.path.exists(os.path.join(MODELS_DIR, "multiclass_model.pkl"))


@pytest.mark.skipif(not MODELS_PRESENT, reason="run src/train.py first")
def test_predictor_output_schema():
    from src.predict import IDSPredictor
    X = np.load(os.path.join(MODELS_DIR, "demo_traffic_sample_scaled.npy"))
    predictor = IDSPredictor()
    results = predictor.predict(X[:5])

    assert len(results) == 5
    for r in results:
        assert isinstance(r["is_attack"], bool)
        assert 0.0 <= r["attack_confidence"] <= 1.0
        assert isinstance(r["category"], str)
        assert 0.0 <= r["category_confidence"] <= 1.0


@pytest.mark.skipif(not MODELS_PRESENT, reason="run src/train.py first")
def test_explainer_returns_top_features():
    from src.explain import IDSExplainer
    X = np.load(os.path.join(MODELS_DIR, "demo_traffic_sample_scaled.npy"))
    explainer = IDSExplainer()
    result = explainer.explain_instance(X[0], top_k=5)

    assert result["prediction"] in ("attack", "normal")
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_features"]) == 5
    for feat in result["top_features"]:
        assert "feature" in feat and "impact" in feat
