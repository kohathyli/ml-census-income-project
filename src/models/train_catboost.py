from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from joblib import dump
from sklearn.model_selection import train_test_split

from src.data.load_data import (
    DEFAULT_COLUMNS_PATH,
    DEFAULT_DATA_PATH,
    load_census_data,
    split_features_target_weights,
)
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import classification_metrics
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost census income classifier.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--columns-path", type=str, default=str(DEFAULT_COLUMNS_PATH))
    parser.add_argument("--output-dir", type=str, default="artifacts/catboost_classifier")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def choose_threshold_grid(
    y_true: pd.Series,
    y_score: np.ndarray,
    sample_weight: pd.Series,
    min_recall: float = 0.30,
) -> tuple[float, dict]:
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.linspace(0.05, 0.95, 181)
    y_true_np = y_true.to_numpy()
    w_np = sample_weight.to_numpy()

    best_thr = 0.5
    best_summary = None
    best_obj = -np.inf

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        precision = precision_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
        recall = recall_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
        f1 = f1_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)

        if recall >= min_recall and precision > best_obj:
            best_obj = precision
            best_thr = float(thr)
            best_summary = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    if best_summary is None:
        best_f1 = -np.inf
        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
            recall = recall_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
            f1 = f1_score(y_true_np, y_pred, sample_weight=w_np, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
                best_summary = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

    return best_thr, best_summary


def evaluate_probs(y_true: pd.Series, y_score: np.ndarray, sample_weight: pd.Series, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    metrics = classification_metrics(
        y_true=y_true.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
        sample_weight=sample_weight.to_numpy(),
    )
    metrics["threshold"] = float(threshold)
    return metrics


def save_roc_curve_figure(y_true, y_score, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CatBoost ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_pr_curve_figure(y_true, y_score, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CatBoost Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_shap_summary_plot(
    model,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 15,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_shap_dependence_plot(
    model,
    X_sample: pd.DataFrame,
    feature_name: str,
    output_path: Path,
    interaction_index: str | None = "auto",
) -> None:
    """
    Save SHAP dependence plot for a given feature.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.dependence_plot(
        ind=feature_name,
        shap_values=shap_values,
        features=X_sample,
        interaction_index=interaction_index,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = load_census_data(
        data_path=args.data_path,
        columns_path=args.columns_path,
        max_rows=args.max_rows,
    )
    X, y, weights = split_features_target_weights(df)

    # Ensure categorical columns are strings for CatBoost
    X = X.copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(
        X, y, weights,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    adjusted_val_size = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_trainval, y_trainval, w_trainval,
        test_size=adjusted_val_size,
        stratify=y_trainval,
        random_state=args.random_state,
    )

    w_train_fit = w_train / w_train.mean()

    cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_cols]

    train_pool = Pool(X_train, y_train, weight=w_train_fit, cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_features_idx)
    test_pool = Pool(X_test, y_test, weight=w_test, cat_features=cat_features_idx)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=args.random_state,
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_score = model.predict_proba(val_pool)[:, 1]
    test_score = model.predict_proba(test_pool)[:, 1]


    selected_threshold, threshold_summary = choose_threshold_grid(
        y_true=y_val,
        y_score=val_score,
        sample_weight=w_val,
        min_recall=0.30,
    )

    selected_mask = test_score >= selected_threshold
    selected_count = int(selected_mask.sum())
    total_count = int(len(test_score))
    selected_pct = selected_count / total_count

    print(f"Selected count: {selected_count}")
    print(f"Total count: {total_count}")
    print(f"Selected % of population: {selected_pct:.4f}")

    val_metrics = evaluate_probs(y_val, val_score, w_val, selected_threshold)
    test_metrics = evaluate_probs(y_test, test_score, w_test, selected_threshold)

        # --- save model performance curves ---
    save_roc_curve_figure(
        y_true=y_test,
        y_score=test_score,
        output_path=output_dir / "test_roc_curve_CatBoost.png",
    )

    save_pr_curve_figure(
        y_true=y_test,
        y_score=test_score,
        output_path=output_dir / "test_pr_curve_CatBoost.png",
    )

    shap_sample_size = min(2000, len(X_test))
    X_shap_sample = X_test.sample(n=shap_sample_size, random_state=args.random_state)

    save_shap_summary_plot(
        model=model,
        X_sample=X_shap_sample,
        output_path=output_dir / "shap_summary_plot_catBoost.png",
        max_display=15,
    )

    # --- save SHAP dependence plots for key features ---
    dependence_features = [
        "age",
        "capital gains",
    ]

    for feature in dependence_features:
        if feature in X_shap_sample.columns:
            save_shap_dependence_plot(
                model=model,
                X_sample=X_shap_sample,
                feature_name=feature,
                output_path=output_dir / f"shap_dependence_{feature.replace(' ', '_')}.png",
            )
        else:
            print(f"[Warning] Feature '{feature}' not found in X_shap_sample.columns; skipping dependence plot.")

    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.get_feature_importance(train_pool),
    }).sort_values("importance", ascending=False)
    importances.head(30).to_csv(output_dir / "feature_importance_top30.csv", index=False)

    save_json(
        {
            "dataset_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "positive_rate_full_data": float(y.mean()),
            "selected_threshold": float(selected_threshold),
            "selected_count_test": selected_count,
            "selected_pct_test": float(selected_pct),
            "threshold_selection": {
                "strategy": "precision_at_min_recall",
                "min_recall": 0.30,
                "validation_threshold_summary": threshold_summary,
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model": "CatBoostClassifier",
        },
        output_dir / "metrics.json",
    )

    dump(model, output_dir / "catboost_model.joblib")
    print("CatBoost classification complete.")
    print(f"Selected threshold = {selected_threshold:.4f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()