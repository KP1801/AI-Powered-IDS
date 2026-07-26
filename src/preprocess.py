"""
preprocess.py
Loads raw NSL-KDD text files and turns them into model-ready feature
matrices, for both:
  - binary classification (normal vs attack)
  - multi-class classification (Normal / DoS / Probe / R2L / U2R)

Saves the fitted encoders/scaler to models/ so the exact same
transformation can be replayed at inference time (train/serve parity).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (COLUMN_NAMES, CATEGORICAL_COLS, DATA_DIR, MODELS_DIR,
                        map_attack_category, ensure_dirs)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMN_NAMES, header=None)
    return df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fits encoders/scaler on TRAIN only, applies to both train and test.
    Returns X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat,
    plus the fitted preprocessing objects.
    """
    ensure_dirs()

    train_df = train_df.drop(columns=["difficulty"])
    test_df = test_df.drop(columns=["difficulty"])

    # --- Targets -----------------------------------------------------
    y_train_cat_str = train_df["label"].apply(map_attack_category)
    y_test_cat_str = test_df["label"].apply(map_attack_category)

    y_train_bin = (y_train_cat_str != "Normal").astype(int)
    y_test_bin = (y_test_cat_str != "Normal").astype(int)

    cat_encoder = LabelEncoder()
    cat_encoder.fit(list(y_train_cat_str.unique()) + ["Unknown"])
    y_train_cat = cat_encoder.transform(y_train_cat_str)
    # test set may contain attack categories unseen in train (by design in NSL-KDD);
    # map anything the encoder has never seen to "Unknown"
    known = set(cat_encoder.classes_)
    y_test_cat_str = y_test_cat_str.apply(lambda v: v if v in known else "Unknown")
    y_test_cat = cat_encoder.transform(y_test_cat_str)

    X_train_raw = train_df.drop(columns=["label"])
    X_test_raw = test_df.drop(columns=["label"])

    # --- Categorical encoding -----------------------------------------
    encoders = {}
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train_raw[col])
        # unseen categories in test -> map to a new "unseen" bucket
        mapping = {cls: i for i, cls in enumerate(le.classes_)}
        X_test[col] = X_test_raw[col].map(mapping).fillna(len(mapping)).astype(int)
        encoders[col] = le

    # --- Scaling --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = list(X_train.columns)

    # Persist everything needed to replay this pipeline at inference time
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "cat_encoders.pkl"))
    joblib.dump(cat_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "X_train_raw": X_train, "X_test_raw": X_test,
        "y_train_bin": y_train_bin.values, "y_test_bin": y_test_bin.values,
        "y_train_cat": y_train_cat, "y_test_cat": y_test_cat,
        "feature_names": feature_names, "cat_encoder": cat_encoder,
    }


def run():
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "NSL-KDD files not found. Run `python data/download_data.py` first."
        )
    train_df = load_raw(train_path)
    test_df = load_raw(test_path)
    data = build_features(train_df, test_df)
    print(f"Train shape: {data['X_train'].shape}, Test shape: {data['X_test'].shape}")
    print(f"Binary class balance (train): "
          f"{np.bincount(data['y_train_bin'])} [normal, attack]")
    return data


if __name__ == "__main__":
    run()
