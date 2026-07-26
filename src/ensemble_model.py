"""
ensemble_model.py
Defines the stacking ensemble used for intrusion detection:

  Base learners (diverse by design, so their errors don't overlap much):
    - Random Forest        -> strong on non-linear tabular splits
    - XGBoost               -> strong on structured/imbalanced data
    - MLP (small neural net) -> catches patterns tree models miss

  Meta-learner:
    - Logistic Regression, trained on the out-of-fold predictions of the
      base learners (via StackingClassifier's internal cross-validation),
      which learns how to best combine the three opinions.

Why stacking instead of a single model: on NSL-KDD, individual models often
disagree specifically on R2L/U2R attacks (which are rare and stealthy).
A meta-learner that sees all three opinions typically improves recall on
exactly these hard-to-catch minority classes, which is the highest-value
target for a real IDS.
"""

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def build_stacking_classifier(n_jobs: int = -1, random_state: int = 42) -> StackingClassifier:
    base_learners = [
        ("random_forest", RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=n_jobs,
            random_state=random_state, class_weight="balanced",
        )),
        ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            n_jobs=n_jobs, random_state=random_state,
            eval_metric="logloss", tree_method="hist",
        )),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200,
            random_state=random_state, early_stopping=True,
        )),
    ]

    meta_learner = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=3,
        n_jobs=n_jobs,
        passthrough=False,
    )
    return clf
