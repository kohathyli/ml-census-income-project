from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = Path("artifacts/segmentation_k6/cluster_assignments.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# k=6 technical clusters -> 5 client-facing personas
PERSONA_MAP = {
    1: "Persona 1: Dependents / non-earners",
    3: "Persona 2: Older low-work / retired households",
    0: "Persona 3: Mainstream working households",
    2: "Persona 3: Mainstream working households",
    5: "Persona 4: Younger working adults",
    4: "Persona 5: Affluent capital-gains niche",
}


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    return float(np.average(x, weights=w))


def weighted_top_category(df: pd.DataFrame, col: str, weight_col: str = "weight") -> str:
    tmp = (
        df.groupby(col, dropna=False)[weight_col]
        .sum()
        .sort_values(ascending=False)
    )
    return str(tmp.index[0])


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # binary outcome
    df["income_gt_50k"] = df["label"].astype(str).str.contains(">50K").astype(int)

    # map technical clusters to client personas
    df["persona"] = df["cluster"].map(PERSONA_MAP)

    if df["persona"].isna().any():
        missing_clusters = sorted(df.loc[df["persona"].isna(), "cluster"].unique())
        raise ValueError(f"Missing persona mapping for clusters: {missing_clusters}")

    total_weight = df["weight"].sum()

    rows = []
    for persona, g in df.groupby("persona"):
        w = g["weight"]

        row = {
            "persona": persona,
            "population_share": w.sum() / total_weight,
            "high_income_rate": weighted_mean(g["income_gt_50k"], w),
            "avg_age": weighted_mean(g["age"], w),
            "avg_weeks_worked": weighted_mean(g["weeks worked in year"], w),
            "avg_capital_gains": weighted_mean(g["capital gains"], w),
            "top_education": weighted_top_category(g, "education"),
            "top_class_of_worker": weighted_top_category(g, "class of worker"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # order personas
    persona_order = [
        "Persona 1: Dependents / non-earners",
        "Persona 2: Older low-work / retired households",
        "Persona 3: Mainstream working households",
        "Persona 4: Younger working adults",
        "Persona 5: Affluent capital-gains niche",
    ]
    summary["persona"] = pd.Categorical(summary["persona"], categories=persona_order, ordered=True)
    summary = summary.sort_values("persona").reset_index(drop=True)

    # format for report-friendly output
    summary["population_share"] = (summary["population_share"] * 100).round(1)
    summary["high_income_rate"] = (summary["high_income_rate"] * 100).round(1)
    summary["avg_age"] = summary["avg_age"].round(1)
    summary["avg_weeks_worked"] = summary["avg_weeks_worked"].round(1)
    summary["avg_capital_gains"] = summary["avg_capital_gains"].round(0)

    summary = summary.rename(
        columns={
            "persona": "Persona",
            "population_share": "Population Share (%)",
            "high_income_rate": "High-Income Rate (%)",
            "avg_age": "Avg Age",
            "avg_weeks_worked": "Avg Weeks Worked",
            "avg_capital_gains": "Avg Capital Gains",
            "top_education": "Top Education",
            "top_class_of_worker": "Top Class of Worker",
        }
    )

    summary.to_csv(OUTPUT_DIR / "persona_summary_table.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_DIR / 'persona_summary_table.csv'}")


if __name__ == "__main__":
    main()