from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.load_data import (
    DEFAULT_COLUMNS_PATH,
    DEFAULT_DATA_PATH,
    load_census_data,
)

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_income_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "label" not in df.columns:
        raise ValueError(f"Expected 'label' column, got columns: {df.columns.tolist()}")

    df["income_binary"] = (
        df["label"].astype(str).str.strip().str.contains(">50K")
    ).astype(int)

    return df

def find_weight_column(df: pd.DataFrame) -> str:
    possible_weight_cols = [
        "weight",
        "instance weight",
        "survey weight",
        "final weight",
    ]

    for col in possible_weight_cols:
        if col in df.columns:
            return col

    raise ValueError(f"Could not find weight column. Available columns: {df.columns.tolist()}")


def save_top_feature_target_correlation(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Compute absolute correlation between numeric features and the income target,
    then save a heatmap of the top correlated features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # remove weight if present
    numeric_cols = [c for c in numeric_cols if c != "weight"]

    if "income_binary" not in numeric_cols:
        numeric_cols.append("income_binary")

    corr_with_target = (
        df[numeric_cols]
        .corr()["income_binary"]
        .drop("income_binary")
        .abs()
        .sort_values(ascending=False)
    )

    top_features = corr_with_target.head(top_n).index.tolist()

    plot_df = corr_with_target.head(top_n).to_frame(name="abs_correlation_with_income")

    plt.figure(figsize=(6, 8))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
    )
    plt.title("Top Features Correlated with High Income")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_target_correlation_heatmap.png", dpi=300)
    plt.close()

    plot_df.to_csv(OUTPUT_DIR / "feature_target_correlation_table.csv")


def save_top_feature_correlation_matrix(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Save a correlation matrix heatmap for the top numeric features associated with income.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "weight"]

    if "income_binary" not in numeric_cols:
        numeric_cols.append("income_binary")

    corr_with_target = (
        df[numeric_cols]
        .corr()["income_binary"]
        .drop("income_binary")
        .abs()
        .sort_values(ascending=False)
    )

    top_features = corr_with_target.head(top_n).index.tolist()

    corr_matrix = df[top_features + ["income_binary"]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.title("Correlation Matrix of Top Features and Income")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_feature_correlation_matrix.png", dpi=300)
    plt.close()

    corr_matrix.to_csv(OUTPUT_DIR / "top_feature_correlation_matrix.csv")


def save_categorical_income_rate_plot(
    df: pd.DataFrame,
    feature: str,
    weight_col: str,
    output_dir: Path,
    top_n: int = 10,
    min_count: int = 200,
) -> None:
    """
    Plot weighted high-income rate by category for a selected categorical feature.
    Only categories with at least min_count records are kept.
    """
    plot_df = df[[feature, "income_binary", weight_col]].copy()
    plot_df[feature] = plot_df[feature].astype(str).str.strip()

    summary = (
        plot_df.groupby(feature)
        .apply(
            lambda g: pd.Series(
                {
                    "record_count": len(g),
                    "weighted_income_rate": np.average(
                        g["income_binary"],
                        weights=g[weight_col]
                    ),
                    "weighted_population": g[weight_col].sum(),
                }
            )
        )
        .reset_index()
    )

    summary = summary[summary["record_count"] >= min_count].copy()
    summary = summary.sort_values("weighted_income_rate", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary,
        x="weighted_income_rate",
        y=feature,
    )
    plt.xlabel("Weighted High-Income Rate")
    plt.ylabel(feature.replace("_", " ").title())
    plt.title(f"Top Categories by High-Income Rate: {feature.replace('_', ' ').title()}")
    plt.xlim(0, min(1.0, summary["weighted_income_rate"].max() * 1.15))
    plt.tight_layout()
    plt.savefig(output_dir / f"{feature}_income_rate.png", dpi=300)
    plt.close()

    summary.to_csv(output_dir / f"{feature}_income_rate_table.csv", index=False)


def save_key_categorical_plots(df: pd.DataFrame, output_dir: Path) -> None:
    weight_col = find_weight_column(df)

    candidate_features = [
        "education",
        "marital stat",
        "class of worker",
        "major occupation code",
        "major industry code",
    ]

    available_features = [col for col in candidate_features if col in df.columns]

    if not available_features:
        raise ValueError(
            f"None of the candidate categorical features were found. Available columns: {df.columns.tolist()}"
        )

    for feature in available_features:
        save_categorical_income_rate_plot(
            df=df,
            feature=feature,
            weight_col=weight_col,
            output_dir=output_dir,
            top_n=10,
            min_count=200,
        )


def main() -> None:
    df = load_census_data(
        data_path=DEFAULT_DATA_PATH,
        columns_path=DEFAULT_COLUMNS_PATH,
        max_rows=None,
    )
    df = make_income_binary(df)

    save_top_feature_target_correlation(df, top_n=15)
    save_top_feature_correlation_matrix(df, top_n=10)

    save_key_categorical_plots(df, OUTPUT_DIR)

    print(f"EDA figures saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()