"""
Train the XGBoost IDS classifier on UNSW-NB15 (primary) or NSL-KDD (secondary
benchmark), exactly following Section 3.3.1 of the base paper:

  - scale_pos_weight = N_normal / N_attack   (native class-imbalance handling)
  - stratified 20% split off the TRAIN set used ONLY for early stopping
  - final metrics reported on the untouched official test set
  - persists model.pkl + scaler.pkl + label_encoders.pkl + feature_columns.pkl

Run:
    python -m src.ids.train --dataset unsw
    python -m src.ids.train --dataset nsl_kdd
"""
import argparse
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

from src.ids.config import (
    MODEL_DIR, UNSW_TRAIN_CSV, UNSW_TEST_CSV, UNSW_ARTIFACT_PREFIX,
    NSL_TRAIN_TXT, NSL_TEST_TXT, NSL_ARTIFACT_PREFIX,
    XGB_PARAMS, VAL_SPLIT, RANDOM_STATE,
)
from src.ids.preprocess import (
    load_unsw, load_unsw_combined_resplit, preprocess_unsw,
    load_nsl_kdd, preprocess_nsl_kdd,
)


def train_and_evaluate(X_train, X_test, y_train, y_test, artifact_prefix: str):
    # scale_pos_weight = N_normal / N_attack  (Eq. 1 in the paper)
    n_normal = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    scale_pos_weight = n_normal / n_attack
    print(f"[{artifact_prefix}] class counts -> normal={n_normal}, attack={n_attack}, "
          f"scale_pos_weight={scale_pos_weight:.4f}")

    # Stratified split OFF the training set, used only for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, stratify=y_train,
        random_state=RANDOM_STATE,
    )

    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"[{artifact_prefix}] trained {model.n_estimators} max trees "
          f"(best_iteration={getattr(model, 'best_iteration', 'n/a')}) "
          f"in {time.time() - t0:.1f}s")

    # Final evaluation on the untouched official test set
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {artifact_prefix} test-set results ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    with open(MODEL_DIR / f"{artifact_prefix}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                    confusion_matrix=cm.tolist())
    return model, metrics


def run(dataset: str):
    if dataset == "unsw":
        train_df, test_df = load_unsw(UNSW_TRAIN_CSV, UNSW_TEST_CSV)
        X_train, X_test, y_train, y_test, feats = preprocess_unsw(
            train_df, test_df, UNSW_ARTIFACT_PREFIX
        )
        train_and_evaluate(X_train, X_test, y_train, y_test, UNSW_ARTIFACT_PREFIX)

    elif dataset == "unsw_resplit":
        # Alternate reproduction matching the paper's stated class counts
        # (see preprocess.load_unsw_combined_resplit docstring)
        train_df, test_df = load_unsw_combined_resplit(UNSW_TRAIN_CSV, UNSW_TEST_CSV)
        X_train, X_test, y_train, y_test, feats = preprocess_unsw(
            train_df, test_df, UNSW_ARTIFACT_PREFIX
        )
        train_and_evaluate(X_train, X_test, y_train, y_test, UNSW_ARTIFACT_PREFIX)

    elif dataset == "nsl_kdd":
        train_df, test_df = load_nsl_kdd(NSL_TRAIN_TXT, NSL_TEST_TXT)
        X_train, X_test, y_train, y_test, feats = preprocess_nsl_kdd(
            train_df, test_df, NSL_ARTIFACT_PREFIX
        )
        train_and_evaluate(X_train, X_test, y_train, y_test, NSL_ARTIFACT_PREFIX)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["unsw", "unsw_resplit", "nsl_kdd"], required=True)
    args = parser.parse_args()
    run(args.dataset)
