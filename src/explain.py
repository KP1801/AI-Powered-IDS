"""
explain.py
Explainability layer for the IDS -- pure Python/NumPy, no native/compiled
dependencies beyond scikit-learn itself.

WHY NOT SHAP: the `shap` library pulls in `numba` for some of its internal
utilities, and numba ships a compiled native extension (a `.pyd`/`.so` file)
that JIT-compiles code at runtime. On locked-down corporate Windows machines
with an Application Control / WDAC policy, loading that native module can be
blocked outright ("ImportError: DLL load failed ... Application Control
policy has blocked this file"), even though the binary itself is completely
legitimate. Since this project should work on locked-down machines too,
explanations here are implemented from scratch using the same well-known
technique behind the "treeinterpreter" package: decompose a Random Forest's
prediction into a bias term (the class-1 probability at the root of each
tree, i.e. before looking at any feature) plus a sum of per-feature
contributions, computed by walking each tree's decision path for a given
instance and attributing the CHANGE in predicted probability at each split
to the feature that split was made on.

This is mathematically exact for a single decision tree:
    bias + sum(contributions) == that tree's predicted probability
and is averaged across every tree in the forest for the final explanation.
"""

import os
import sys
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import MODELS_DIR


class IDSExplainer:
    def __init__(self):
        self.binary_model = joblib.load(os.path.join(MODELS_DIR, "binary_model.pkl"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        # Pull out the fitted Random Forest base-learner from the stack --
        # it's the most naturally interpretable of the three base learners.
        self.rf = dict(self.binary_model.named_estimators_)["random_forest"]

    @staticmethod
    def _node_proba_class1(tree, node_id):
        """Fraction of class-1 (attack) samples at a given tree node."""
        value = tree.value[node_id][0]  # shape (n_classes,), weighted counts
        total = value.sum()
        return float(value[1] / total) if total > 0 else 0.0

    def _tree_contributions(self, tree, x_row):
        """Walk one tree's decision path for x_row, returning (bias, contributions)."""
        contributions = np.zeros(len(self.feature_names))
        node = 0
        bias = self._node_proba_class1(tree, 0)

        # A node is a leaf when its two children are the same sentinel value
        while tree.children_left[node] != tree.children_right[node]:
            feat = tree.feature[node]
            threshold = tree.threshold[node]
            go_left = x_row[feat] <= threshold
            next_node = tree.children_left[node] if go_left else tree.children_right[node]

            parent_p = self._node_proba_class1(tree, node)
            child_p = self._node_proba_class1(tree, next_node)
            contributions[feat] += (child_p - parent_p)
            node = next_node

        return bias, contributions

    def explain_instance(self, x_row: np.ndarray, top_k: int = 5):
        """
        x_row: a single scaled feature vector (1D array, already preprocessed)
        Returns the model's prediction/confidence plus the top_k features
        (by absolute contribution) that pushed the prediction towards
        "attack" (positive impact) or "normal" (negative impact).
        """
        x_row = np.asarray(x_row).reshape(-1)

        pred = self.binary_model.predict(x_row.reshape(1, -1))[0]
        proba = self.binary_model.predict_proba(x_row.reshape(1, -1))[0][pred]

        # Average bias + per-feature contributions across every tree
        all_contribs = np.zeros(len(self.feature_names))
        all_bias = 0.0
        for est in self.rf.estimators_:
            bias, contribs = self._tree_contributions(est.tree_, x_row)
            all_bias += bias
            all_contribs += contribs
        n_trees = len(self.rf.estimators_)
        all_bias /= n_trees
        all_contribs /= n_trees

        pairs = list(zip(self.feature_names, all_contribs))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top_features = pairs[:top_k]

        return {
            "prediction": "attack" if pred == 1 else "normal",
            "confidence": float(proba),
            "top_features": [
                {"feature": f, "impact": float(v)} for f, v in top_features
            ],
        }

    def explain_batch(self, X: np.ndarray, top_k: int = 5):
        return [self.explain_instance(row, top_k) for row in X]


if __name__ == "__main__":
    # Small demo using the saved sample traffic from training
    X_sample = np.load(os.path.join(MODELS_DIR, "demo_traffic_sample_scaled.npy"))
    explainer = IDSExplainer()
    result = explainer.explain_instance(X_sample[0])
    print(result)
