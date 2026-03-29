from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import (
    DEFAULT_COLUMNS_PATH,
    DEFAULT_DATA_PATH,
    load_census_data,
    split_features_target_weights,
)
from src.data.preprocess import build_preprocessor
from src.utils.io import ensure_dir, save_json, save_text
from src.utils.metrics import (
    classification_metrics,
    classification_report_text,
    weighted_confusion_matrix,
)
from src.utils.visualization import (
    save_confusion_matrix_figure,
    save_pr_curve,
    save_roc_curve,
)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate census income classifier.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--columns-path", type=str, default=str(DEFAULT_COLUMNS_PATH))
    parser.add_argument("--output-dir", type=str, default="artifacts/classifier")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def compute_feature_importance(pipeline: Pipeline, output_path: Path) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_.ravel()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    importance.head(30).drop(columns=["abs_coefficient"]).to_csv(output_path, index=False)


def choose_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    sample_weight: pd.Series,
    strategy: str = "precision_at_min_recall",
    min_recall: float = 0.30,
) -> tuple[float, dict]:
    from sklearn.metrics import precision_score, recall_score, f1_score

    candidate_thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_summary = None
    best_objective = -np.inf

    y_true_np = y_true.to_numpy()
    w_np = sample_weight.to_numpy()

    for thr in candidate_thresholds:
        y_pred = (y_score >= thr).astype(int)

        precision = precision_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
        recall = recall_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
        f1 = f1_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)

        if strategy == "f1":
            objective = f1
        elif strategy == "precision_at_min_recall":
            if recall >= min_recall:
                objective = precision
            else:
                objective = -np.inf
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        if objective > best_objective:
            best_objective = objective
            best_threshold = float(thr)
            best_summary = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    if best_summary is None:
        # fallback: best F1 if no threshold satisfies min recall
        best_threshold = 0.5
        best_f1 = -np.inf
        for thr in candidate_thresholds:
            y_pred = (y_score >= thr).astype(int)
            precision = precision_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
            recall = recall_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
            f1 = f1_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thr)
                best_summary = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

    return best_threshold, best_summary


def save_roc_data(
    y_true: pd.Series,
    y_score: np.ndarray,
    output_path: Path,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true.to_numpy(), y_score)
    roc_auc = auc(fpr, tpr)

    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    roc_df.to_csv(output_path, index=False)

    return float(roc_auc)

def evaluate_split(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    output_dir: Path,
    threshold: float,
) -> dict:
    y_score = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = classification_metrics(
        y_true=y.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
        sample_weight=weights.to_numpy(),
    )
    metrics["threshold"] = float(threshold)

    report = classification_report_text(
        y_true=y.to_numpy(),
        y_pred=y_pred,
        sample_weight=weights.to_numpy(),
    )
    save_text(report, output_dir / f"{name}_classification_report.txt")

    cm = weighted_confusion_matrix(
        y_true=y.to_numpy(),
        y_pred=y_pred,
        sample_weight=weights.to_numpy(),
    )
    save_confusion_matrix_figure(
        cm=cm,
        output_path=output_dir / f"{name}_confusion_matrix.png",
        title=f"{name.title()} Weighted Confusion Matrix (threshold={threshold:.3f})",
    )
    save_roc_curve(
        y,
        y_score,
        output_dir / f"{name}_roc_curve.png",
        title=f"Logistic Regression ROC Curve ({name.title()} Set)"
    )

    roc_auc_value = save_roc_data(
        y_true=y,
        y_score=y_score,
        output_path=output_dir / f"{name}_roc_curve_data.csv",
    )
    save_pr_curve(y, y_score, output_dir / f"{name}_pr_curve.png")

    metrics["roc_auc_recomputed"] = roc_auc_value
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = load_census_data(
        data_path=args.data_path,
        columns_path=args.columns_path,
        max_rows=args.max_rows,
    )
    X, y, weights = split_features_target_weights(df)

    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(
        X,
        y,
        weights,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    adjusted_val_size = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_trainval,
        y_trainval,
        w_trainval,
        test_size=adjusted_val_size,
        stratify=y_trainval,
        random_state=args.random_state,
    )
    w_train_fit = w_train / w_train.mean()

    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=0.2,
    max_iter=30000,
    tol=1e-3,
    random_state=args.random_state,
)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train, model__sample_weight=w_train_fit.to_numpy())

    # --- choose threshold on validation set ---
    val_scores = pipeline.predict_proba(X_val)[:, 1]
    selected_threshold, threshold_summary = choose_threshold(
        y_true=y_val,
        y_score=val_scores,
        sample_weight=w_val,
        strategy="precision_at_min_recall",
        min_recall=0.30,
    )

    val_metrics = evaluate_split(
        "validation",
        pipeline,
        X_val,
        y_val,
        w_val,
        output_dir,
        threshold=selected_threshold,
    )
    test_metrics = evaluate_split(
        "test",
        pipeline,
        X_test,
        y_test,
        w_test,
        output_dir,
        threshold=selected_threshold,
    )

    save_json(
        {
            "dataset_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "positive_rate_full_data": float(y.mean()),
            "selected_threshold": float(selected_threshold),
            "threshold_selection": {
                "strategy": "precision_at_min_recall",
                "min_recall": 0.30,
                "validation_threshold_summary": threshold_summary,
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model": "Weighted Logistic Regression with One-Hot Encoding",
        },
        output_dir / "metrics.json",
    )

    compute_feature_importance(pipeline, output_dir / "feature_importance_top30.csv")
    dump(pipeline, output_dir / "classifier_pipeline.joblib")

    print("Classification complete.")
    print(f"Selected threshold = {selected_threshold:.4f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()