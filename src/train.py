"""
train.py
End-to-end training entry point.

Trains TWO stacking ensembles:
  1. Binary model   -> "is this connection malicious at all?" (fast first pass)
  2. Multi-class model -> "which category of attack is it?" (DoS/Probe/R2L/U2R)

This two-stage design mirrors how real IDS pipelines are often built:
a cheap/fast filter first, then a more detailed classification only on
what got flagged -- and it also gives the dashboard richer alerts
("DoS attack, 98% confidence" instead of just "attack").

Usage:
    python src/train.py
"""

import os
import sys
import time
import json

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run as preprocess_run
from src.ensemble_model import build_stacking_classifier
from src.utils import MODELS_DIR, ensure_dirs, CATEGORY_LIST


def train_and_eval(X_train, y_train, X_test, y_test, label_names, model_name):
    print(f"\n{'='*60}\nTraining {model_name} stacking ensemble...\n{'='*60}")
    t0 = time.time()
    clf = build_stacking_classifier()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Trained in {elapsed:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    metrics = {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "train_seconds": elapsed, "classification_report": report,
        "confusion_matrix": cm, "labels": label_names,
    }
    return clf, metrics


def main():
    ensure_dirs()
    data = preprocess_run()

    # --- Binary model: normal vs attack ---------------------------------
    bin_clf, bin_metrics = train_and_eval(
        data["X_train"], data["y_train_bin"],
        data["X_test"], data["y_test_bin"],
        label_names=["normal", "attack"], model_name="BINARY (normal vs attack)",
    )
    joblib.dump(bin_clf, os.path.join(MODELS_DIR, "binary_model.pkl"))

    # --- Multi-class model: attack category ------------------------------
    cat_encoder = data["cat_encoder"]
    present_labels = sorted(set(data["y_train_cat"]) | set(data["y_test_cat"]))
    label_names = [cat_encoder.classes_[i] for i in present_labels]

    cat_clf, cat_metrics = train_and_eval(
        data["X_train"], data["y_train_cat"],
        data["X_test"], data["y_test_cat"],
        label_names=label_names, model_name="MULTI-CLASS (attack category)",
    )
    joblib.dump(cat_clf, os.path.join(MODELS_DIR, "multiclass_model.pkl"))

    # --- Persist a small sample of raw test rows for the dashboard demo --
    sample_idx = np.random.default_rng(0).choice(
        len(data["X_test_raw"]), size=min(500, len(data["X_test_raw"])), replace=False
    )
    data["X_test_raw"].iloc[sample_idx].to_csv(
        os.path.join(MODELS_DIR, "demo_traffic_sample.csv"), index=False
    )
    np.save(os.path.join(MODELS_DIR, "demo_traffic_sample_scaled.npy"),
            data["X_test"][sample_idx])
    np.save(os.path.join(MODELS_DIR, "demo_traffic_sample_ybin.npy"),
            data["y_test_bin"][sample_idx])
    np.save(os.path.join(MODELS_DIR, "demo_traffic_sample_ycat.npy"),
            data["y_test_cat"][sample_idx])

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"binary": bin_metrics, "multiclass": cat_metrics}, f, indent=2)

    print("\nAll models and metrics saved to models/")


if __name__ == "__main__":
    main()
