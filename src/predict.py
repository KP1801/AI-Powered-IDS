"""
predict.py
Simple prediction interface wrapping the two trained stacking ensembles.
Used by the Streamlit dashboard and can also be used standalone/CLI.
"""

import os
import sys
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import MODELS_DIR


class IDSPredictor:
    def __init__(self):
        self.binary_model = joblib.load(os.path.join(MODELS_DIR, "binary_model.pkl"))
        self.multiclass_model = joblib.load(os.path.join(MODELS_DIR, "multiclass_model.pkl"))
        self.label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    def predict(self, X: np.ndarray):
        """
        X: 2D scaled feature array (n_samples, n_features)
        Returns a list of dicts, one per row:
          is_attack, attack_confidence, category, category_confidence
        """
        bin_pred = self.binary_model.predict(X)
        bin_proba = self.binary_model.predict_proba(X)

        cat_pred = self.multiclass_model.predict(X)
        cat_proba = self.multiclass_model.predict_proba(X)
        cat_labels = self.label_encoder.inverse_transform(cat_pred)

        results = []
        for i in range(len(X)):
            results.append({
                "is_attack": bool(bin_pred[i]),
                "attack_confidence": float(bin_proba[i][bin_pred[i]]),
                "category": str(cat_labels[i]),
                "category_confidence": float(cat_proba[i][cat_pred[i]]),
            })
        return results
