"""
adversarial_test.py
Adversarial robustness evaluation for the IDS.

A real attacker doesn't just send raw attack traffic -- they may try to
tweak connection features (e.g. padding packet sizes, adjusting timing,
changing which service/flag combination they use) specifically to slip
past a known detector. This script simulates that by taking KNOWN attack
traffic and applying small, bounded perturbations to a subset of
"attacker-controllable" features, then measuring how much the model's
detection rate drops.

This is a black-box, gradient-free perturbation search (random + greedy
hill-climbing within bounds) rather than a white-box gradient attack,
because the ensemble includes non-differentiable tree models -- this
mirrors a realistic attacker who can only query the system, not see its
internals.

Usage:
    python src/adversarial_test.py
"""

import os
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run as preprocess_run
from src.utils import MODELS_DIR

# Features a real attacker could plausibly manipulate without changing the
# fundamental nature of the attack (e.g. padding bytes, tweaking timing).
# Indices are resolved dynamically from feature_names below.
PERTURBABLE_FEATURES = [
    "src_bytes", "dst_bytes", "duration", "wrong_fragment",
    "count", "srv_count", "same_srv_rate", "diff_srv_rate",
]

MAX_PERTURBATION_STD = 1.5   # bounded perturbation, in units of scaled std-dev
N_TRIALS_PER_SAMPLE = 25     # random search budget per sample


def evade_search(model, x, feature_idx, rng):
    """Random search within a bounded L-infinity ball for a perturbation
    that flips the model's prediction from attack -> normal."""
    best_x = x.copy()
    original_pred = model.predict(x.reshape(1, -1))[0]
    if original_pred == 0:
        return best_x, False  # already predicted normal, nothing to evade

    for _ in range(N_TRIALS_PER_SAMPLE):
        candidate = x.copy()
        for idx in feature_idx:
            delta = rng.uniform(-MAX_PERTURBATION_STD, MAX_PERTURBATION_STD)
            candidate[idx] += delta
        pred = model.predict(candidate.reshape(1, -1))[0]
        if pred == 0:
            return candidate, True
    return best_x, False


def main():
    print("Loading trained binary model and test data...")
    binary_model = joblib.load(os.path.join(MODELS_DIR, "binary_model.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    data = preprocess_run()

    X_test, y_test = data["X_test"], data["y_test_bin"]
    attack_idx = np.where(y_test == 1)[0]

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(attack_idx, size=min(300, len(attack_idx)), replace=False)

    feature_idx = [feature_names.index(f) for f in PERTURBABLE_FEATURES
                   if f in feature_names]

    print(f"Testing {len(sample_idx)} known-attack samples, perturbing "
          f"{len(feature_idx)} attacker-controllable features "
          f"(max {MAX_PERTURBATION_STD} std-devs per feature, "
          f"{N_TRIALS_PER_SAMPLE} random tries each)...\n")

    baseline_caught = 0
    evaded = 0
    for i in sample_idx:
        x = X_test[i]
        pred = binary_model.predict(x.reshape(1, -1))[0]
        if pred == 1:
            baseline_caught += 1
            _, success = evade_search(binary_model, x, feature_idx, rng)
            if success:
                evaded += 1

    baseline_recall = baseline_caught / len(sample_idx)
    post_attack_recall = (baseline_caught - evaded) / len(sample_idx)
    evasion_rate = evaded / baseline_caught if baseline_caught else 0.0

    print("=" * 60)
    print("ADVERSARIAL ROBUSTNESS REPORT")
    print("=" * 60)
    print(f"Attack samples tested:            {len(sample_idx)}")
    print(f"Caught by model (baseline):       {baseline_caught} "
          f"({baseline_recall*100:.1f}% recall)")
    print(f"Successfully evaded via bounded")
    print(f"  feature perturbation:           {evaded} "
          f"({evasion_rate*100:.1f}% of caught samples)")
    print(f"Recall after adversarial attempt: {post_attack_recall*100:.1f}%")
    print()
    if evasion_rate > 0.15:
        print("⚠  Non-trivial evasion rate: consider adversarial training "
              "(augmenting the training set with perturbed samples) or "
              "input-space anomaly detection as a second-stage check.")
    else:
        print("✓ Model shows reasonable robustness to small, bounded "
              "feature-space perturbations of this kind.")


if __name__ == "__main__":
    main()
