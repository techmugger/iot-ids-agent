"""
Preprocessing pipeline for UNSW-NB15 and NSL-KDD.

Mirrors Section 3.3.1 of the base paper:
  - drop ID / non-predictive columns
  - label-encode categorical features (persisted to label_encoders.pkl)
  - scale numeric features (persisted to scaler.pkl)
  - persist final feature column order (feature_columns.pkl) so live
    inference can align single samples to the exact training schema
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.ids.config import MODEL_DIR, NSL_KDD_COLUMNS


def _save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# UNSW-NB15
# --------------------------------------------------------------------------
def load_unsw(train_csv: Path, test_csv: Path):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    return train, test


def load_unsw_combined_resplit(train_csv: Path, test_csv: Path, random_state: int = 42):
    """
    Alternate loader: pools the official train+test partitions and performs a
    fresh stratified split sized to match the paper's stated 82,332-sample
    test set. The base paper's Section 3.3.1 class counts (93,000 normal /
    82,341 attack in "the training set") don't match the official UNSW-NB15
    train partition (56,000 / 119,341) -- they match the *combined* dataset's
    normal count instead, suggesting the authors likely pooled-and-resplit
    rather than using the official partition. This loader reproduces that.
    """
    from sklearn.model_selection import train_test_split

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    combined = pd.concat([train, test], ignore_index=True)

    test_frac = len(test) / len(combined)
    tr, te = train_test_split(
        combined, test_size=test_frac, stratify=combined["label"],
        random_state=random_state,
    )
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def preprocess_unsw(train_df: pd.DataFrame, test_df: pd.DataFrame, artifact_prefix: str):
    """
    Returns X_train, X_test, y_train, y_test (binary: 0=Normal, 1=Attack)
    and persists scaler / encoders / feature_columns to MODEL_DIR.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop identifier + the multi-class label (we do binary classification,
    # matching the paper's reported metrics); 'label' is already 0/1 in UNSW-NB15.
    drop_cols = [c for c in ["id", "attack_cat"] if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    y_train = train_df.pop("label").astype(int).values
    y_test = test_df.pop("label").astype(int).values

    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in train_df.columns if c not in categorical_cols]

    # Label-encode categoricals; fit ONLY on train, map unseen test categories
    # to a reserved "unknown" bucket instead of crashing.
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le

        known = set(le.classes_)
        test_df[col] = test_df[col].astype(str).apply(lambda v: v if v in known else "__unknown__")
        # extend encoder to handle unseen label deterministically
        if "__unknown__" not in known:
            le.classes_ = np.append(le.classes_, "__unknown__")
        test_df[col] = le.transform(test_df[col])

    # Scale numeric columns
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    feature_columns = train_df.columns.tolist()

    _save_pickle(encoders, MODEL_DIR / f"{artifact_prefix}_label_encoders.pkl")
    _save_pickle(scaler, MODEL_DIR / f"{artifact_prefix}_scaler.pkl")
    _save_pickle(feature_columns, MODEL_DIR / f"{artifact_prefix}_feature_columns.pkl")

    return train_df.values, test_df.values, y_train, y_test, feature_columns


# --------------------------------------------------------------------------
# NSL-KDD  (independent 41-feature schema, no cross-mapping to UNSW-NB15)
# --------------------------------------------------------------------------
def load_nsl_kdd(train_txt: Path, test_txt: Path):
    train = pd.read_csv(train_txt, names=NSL_KDD_COLUMNS, header=None)
    test = pd.read_csv(test_txt, names=NSL_KDD_COLUMNS, header=None)
    return train, test


def preprocess_nsl_kdd(train_df: pd.DataFrame, test_df: pd.DataFrame, artifact_prefix: str):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 'difficulty' is a KDD-specific scoring column, not a feature -> drop.
    for df in (train_df, test_df):
        if "difficulty" in df.columns:
            df.drop(columns=["difficulty"], inplace=True)

    # Binary label: 'normal' -> 0, anything else (dos/probe/r2l/u2r/*) -> 1
    y_train = (train_df.pop("label") != "normal").astype(int).values
    y_test = (test_df.pop("label") != "normal").astype(int).values

    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in train_df.columns if c not in categorical_cols]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le

        known = set(le.classes_)
        test_df[col] = test_df[col].astype(str).apply(lambda v: v if v in known else "__unknown__")
        if "__unknown__" not in known:
            le.classes_ = np.append(le.classes_, "__unknown__")
        test_df[col] = le.transform(test_df[col])

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    feature_columns = train_df.columns.tolist()

    _save_pickle(encoders, MODEL_DIR / f"{artifact_prefix}_label_encoders.pkl")
    _save_pickle(scaler, MODEL_DIR / f"{artifact_prefix}_scaler.pkl")
    _save_pickle(feature_columns, MODEL_DIR / f"{artifact_prefix}_feature_columns.pkl")

    return train_df.values, test_df.values, y_train, y_test, feature_columns


def load_artifacts(artifact_prefix: str):
    """Load persisted encoders/scaler/feature_columns/model for live inference."""
    encoders = _load_pickle(MODEL_DIR / f"{artifact_prefix}_label_encoders.pkl")
    scaler = _load_pickle(MODEL_DIR / f"{artifact_prefix}_scaler.pkl")
    feature_columns = _load_pickle(MODEL_DIR / f"{artifact_prefix}_feature_columns.pkl")
    model = _load_pickle(MODEL_DIR / f"{artifact_prefix}_model.pkl")
    return model, scaler, encoders, feature_columns
