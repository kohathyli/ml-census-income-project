from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data.load_data import (
    DEFAULT_COLUMNS_PATH,
    DEFAULT_DATA_PATH,
    load_census_data,
    split_features_target_weights,
)


MODEL_PATH = Path("artifacts/catboost_classifier/catboost_model.joblib")
METRICS_PATH = Path("artifacts/catboost_classifier/metrics.json")
OUTPUT_PATH = Path("reports/figures/threshold_scenarios.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

THRESHOLDS = [0.70, 0.50, 0.30]

COST_PER_CONTACT = 2.0

def prepare_catboost_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    return X


def load_selected_threshold() -> float:
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return float(metrics["selected_threshold"])


def build_test_split():
    df = load_census_data(
        data_path=DEFAULT_DATA_PATH,
        columns_path=DEFAULT_COLUMNS_PATH,
        max_rows=None,
    )
    X, y, weights = split_features_target_weights(df)

    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(
        X, y, weights,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    adjusted_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_trainval, y_trainval, w_trainval,
        test_size=adjusted_val_size,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    return X_test, y_test, w_test


def evaluate_thresholds(
    y_true: pd.Series,
    y_score: np.ndarray,
    thresholds: list[float],
    cost_per_contact: float = 2.0,
) -> pd.DataFrame:
    rows = []
    y_true_np = y_true.to_numpy()

    total_count = len(y_true_np)

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)

        precision = precision_score(y_true_np, y_pred, zero_division=0)
        recall = recall_score(y_true_np, y_pred, zero_division=0)
        f1 = f1_score(y_true_np, y_pred, zero_division=0)

        selected_count = int(y_pred.sum())
        selected_pct = selected_count / total_count
        est_cost = selected_count * cost_per_contact

        rows.append(
            {
                "Threshold": round(thr, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3),
                "% Selected": round(selected_pct * 100, 2),
                "Selected Count": selected_count,
                "Estimated Cost": round(est_cost, 2),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    model = load(MODEL_PATH)

    X_test, y_test, w_test = build_test_split()
    X_test = prepare_catboost_features(X_test)

    y_score = model.predict_proba(X_test)[:, 1]

    selected_threshold = load_selected_threshold()
    threshold_list = sorted(set(THRESHOLDS + [selected_threshold]), reverse=True)

    table = evaluate_thresholds(
        y_true=y_test,
        y_score=y_score,
        thresholds=threshold_list,
        cost_per_contact=COST_PER_CONTACT,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()