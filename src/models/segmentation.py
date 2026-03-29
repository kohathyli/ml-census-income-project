from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.data.load_data import (
    DEFAULT_COLUMNS_PATH,
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    load_census_data,
)
from src.data.preprocess import build_preprocessor, get_feature_types
from src.utils.io import ensure_dir, save_text
from src.utils.visualization import save_line_plot, save_scatter_plot


PROFILE_CATEGORICAL_COLUMNS = [
    "education",
    "marital stat",
    "race",
    "sex",
    "class of worker",
    "major industry code",
    "major occupation code",
    "citizenship",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build segmentation model for census data.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--columns-path", type=str, default=str(DEFAULT_COLUMNS_PATH))
    parser.add_argument("--output-dir", type=str, default="artifacts/segmentation")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--svd-components", type=int, default=30)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--selection-sample-size", type=int, default=30000)
    parser.add_argument("--scatter-sample-size", type=int, default=8000)
    return parser.parse_args()



def choose_k(
    X_reduced: np.ndarray,
    random_state: int,
    k_min: int,
    k_max: int,
    selection_sample_size: int,
) -> tuple[int, pd.DataFrame]:
    n = X_reduced.shape[0]
    sample_n = min(selection_sample_size, n)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n, size=sample_n, replace=False) if sample_n < n else np.arange(n)
    X_sample = X_reduced[sample_idx]

    rows = []
    best_k = None
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        rows.append({"k": k, "silhouette_score": float(score)})
        if score > best_score:
            best_score = score
            best_k = k

    results = pd.DataFrame(rows)
    assert best_k is not None
    return best_k, results



def weighted_cluster_sizes(labels: np.ndarray, weights: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"cluster": labels, "weight": weights.to_numpy()})
    summary = df.groupby("cluster", as_index=False).agg(
        weighted_population=("weight", "sum"),
        record_count=("weight", "size"),
    )
    summary["weighted_population_share"] = (
        summary["weighted_population"] / summary["weighted_population"].sum()
    )
    return summary.sort_values("cluster").reset_index(drop=True)



def numeric_cluster_profile(
    data: pd.DataFrame,
    labels: np.ndarray,
    weights: pd.Series,
) -> pd.DataFrame:
    numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    profile_rows = []

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_data = data.loc[mask, numeric_cols]
        cluster_weights = weights.loc[mask].to_numpy()

        row = {"cluster": int(cluster_id)}
        for col in numeric_cols:
            values = cluster_data[col].astype(float).fillna(cluster_data[col].median()).to_numpy()
            row[col] = float(np.average(values, weights=cluster_weights))
        profile_rows.append(row)

    return pd.DataFrame(profile_rows)



def top_categories_for_cluster(
    data: pd.DataFrame,
    labels: np.ndarray,
    weights: pd.Series,
    categorical_columns: list[str],
    top_n: int = 3,
) -> pd.DataFrame:
    rows = []

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_data = data.loc[mask].copy()
        cluster_weights = weights.loc[mask].to_numpy()

        for col in categorical_columns:
            if col not in cluster_data.columns:
                continue
            tmp = pd.DataFrame(
                {
                    "value": cluster_data[col].fillna("Missing").astype(str).to_numpy(),
                    "weight": cluster_weights,
                }
            )
            grouped = (
                tmp.groupby("value", as_index=False)["weight"]
                .sum()
                .sort_values("weight", ascending=False)
                .head(top_n)
            )
            total_weight = grouped["weight"].sum()
            for rank, (_, r) in enumerate(grouped.iterrows(), start=1):
                rows.append(
                    {
                        "cluster": int(cluster_id),
                        "column": col,
                        "rank": rank,
                        "category": r["value"],
                        "weighted_count": float(r["weight"]),
                        "share_within_topn": float(r["weight"] / total_weight) if total_weight else 0.0,
                    }
                )

    return pd.DataFrame(rows)



def build_marketing_summary(
    size_summary: pd.DataFrame,
    numeric_profile: pd.DataFrame,
    top_categories: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = ["# Segment Marketing Summary", ""]

    for _, size_row in size_summary.sort_values("cluster").iterrows():
        cluster_id = int(size_row["cluster"])
        lines.append(f"## Cluster {cluster_id}")
        lines.append(
            f"- Weighted population share: {size_row['weighted_population_share']:.2%}"
        )

        num_row = numeric_profile[numeric_profile["cluster"] == cluster_id].iloc[0]
        lines.append(
            "- Numeric profile highlights: "
            f"average age={num_row.get('age', float('nan')):.1f}, "
            f"average wage per hour={num_row.get('wage per hour', float('nan')):.1f}, "
            f"average weeks worked in year={num_row.get('weeks worked in year', float('nan')):.1f}, "
            f"average capital gains={num_row.get('capital gains', float('nan')):.1f}"
        )

        subset = top_categories[top_categories["cluster"] == cluster_id]
        for col in subset["column"].unique():
            col_subset = subset[subset["column"] == col].sort_values("rank")
            top_values = ", ".join(col_subset["category"].astype(str).tolist())
            lines.append(f"- Top {col}: {top_values}")

        lines.append(
            "- Suggested marketing usage: tailor messaging, pricing, and product mix to the "
            "economic profile and life stage represented by this segment."
        )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = load_census_data(
        data_path=args.data_path,
        columns_path=args.columns_path,
        max_rows=args.max_rows,
    )

    weights = df[WEIGHT_COLUMN].astype(float)
    X = df.drop(columns=[TARGET_COLUMN, WEIGHT_COLUMN])

    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    n_components = min(args.svd_components, max(2, X_transformed.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=args.random_state)
    X_reduced = svd.fit_transform(X_transformed)

    best_k, k_results = choose_k(
        X_reduced=X_reduced,
        random_state=args.random_state,
        k_min=args.k_min,
        k_max=args.k_max,
        selection_sample_size=args.selection_sample_size,
    )
    k_results.to_csv(output_dir / "k_selection.csv", index=False)
    save_line_plot(
        x=k_results["k"],
        y=k_results["silhouette_score"],
        output_path=output_dir / "silhouette_by_k.png",
        title="Silhouette Score by Number of Clusters",
        xlabel="Number of Clusters (k)",
        ylabel="Silhouette Score",
    )

    cluster_model = KMeans(n_clusters=best_k, n_init=10, random_state=args.random_state)
    labels = cluster_model.fit_predict(X_reduced, sample_weight=weights.to_numpy())

    assignments = df.copy()
    assignments["cluster"] = labels
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)

    size_summary = weighted_cluster_sizes(labels, weights)
    size_summary.to_csv(output_dir / "cluster_size_summary.csv", index=False)

    numeric_profile = numeric_cluster_profile(X, labels, weights)
    numeric_profile.to_csv(output_dir / "cluster_numeric_profile.csv", index=False)

    categorical_cols = [c for c in PROFILE_CATEGORICAL_COLUMNS if c in X.columns]
    top_categories = top_categories_for_cluster(X, labels, weights, categorical_cols)
    top_categories.to_csv(output_dir / "cluster_top_categories.csv", index=False)

    build_marketing_summary(
        size_summary=size_summary,
        numeric_profile=numeric_profile,
        top_categories=top_categories,
        output_path=output_dir / "segment_marketing_summary.md",
    )

    rng = np.random.default_rng(args.random_state)
    scatter_n = min(args.scatter_sample_size, len(X_reduced))
    scatter_idx = (
        rng.choice(len(X_reduced), size=scatter_n, replace=False)
        if scatter_n < len(X_reduced)
        else np.arange(len(X_reduced))
    )
    save_scatter_plot(
        x=X_reduced[scatter_idx, 0],
        y=X_reduced[scatter_idx, 1],
        labels=labels[scatter_idx],
        output_path=output_dir / "svd_scatter_clusters.png",
        title="Cluster Scatter Plot (First Two SVD Components)",
    )

    pipeline = {
        "preprocessor": preprocessor,
        "svd": svd,
        "cluster_model": cluster_model,
        "best_k": best_k,
    }
    dump(pipeline, output_dir / "segmentation_pipeline.joblib")

    print("Segmentation complete.")
    print(f"Selected k = {best_k}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
