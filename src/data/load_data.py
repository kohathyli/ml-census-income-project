from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/census-bureau.data")
DEFAULT_COLUMNS_PATH = Path("data/raw/census-bureau.columns")
TARGET_COLUMN = "label"
WEIGHT_COLUMN = "weight"


def load_column_names(columns_path: str | Path = DEFAULT_COLUMNS_PATH) -> list[str]:
    columns_path = Path(columns_path)
    columns = [line.strip() for line in columns_path.read_text(encoding="utf-8").splitlines()]
    columns = [c for c in columns if c]
    return columns


def load_census_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    columns_path: str | Path = DEFAULT_COLUMNS_PATH,
    max_rows: int | None = None,
) -> pd.DataFrame:
    columns = load_column_names(columns_path)

    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=["?"],
        skipinitialspace=True,
        nrows=max_rows,
    )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .replace({"- 50000.": "<=50K", "50000+.": ">50K"})
    )

    return df



def split_features_target_weights(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    weight_col: str = WEIGHT_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col, weight_col])
    y = (df[target_col] == ">50K").astype(int)
    weights = df[weight_col].astype(float)
    return X, y, weights
