
"""
Preprocessing utilities for the UNSW-NB15 dataset.

This module prepares network-flow records
for Agent 1 classification models.
"""

import pandas as pd

from src.data.dataset_loader import COLUMNS


# Columns that must not be used as model features.
EXCLUDED_FEATURES = [
    "Label",
    "attack_cat",
    "srcip",
    "dstip",
    "Stime",
    "Ltime",
]



def get_feature_columns():
    """Return the columns available for model training."""

    return [
        column
        for column in COLUMNS
        if column not in EXCLUDED_FEATURES
    ]


def convert_port_column(series: pd.Series) -> pd.Series:
    """
    Convert decimal and hexadecimal ports to numeric values.

    Missing, invalid, and out-of-range ports become NaN.
    """

    values = series.astype("string").str.strip()

    numeric = pd.to_numeric(values, errors="coerce")

    hex_mask = values.str.match(
        r"(?i)^0x[0-9a-f]+$",
        na=False
    )

    numeric.loc[hex_mask] = (
        values.loc[hex_mask]
        .map(lambda value: int(value, 16))
    )

    valid_mask = (
        numeric.between(0, 65535)
        & (numeric % 1 == 0)
    )

    numeric = numeric.where(valid_mask)

    return numeric.astype("float64")

def clean_port_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean source and destination port columns."""

    required_columns = ["sport", "dsport"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    df = df.copy()

    df["sport"] = convert_port_column(df["sport"])
    df["dsport"] = convert_port_column(df["dsport"])

    return df

CATEGORICAL_COLUMNS = [
    "proto",
    "state",
    "service",
]


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical feature values."""

    df = df.copy()

    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    return df

def validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the binary classification target."""

    df = df.copy()

    if "Label" not in df.columns:
        raise ValueError("Missing target column: Label")

    if df["Label"].isna().any():
        raise ValueError("Target column contains missing values")

    if not df["Label"].isin([0, 1]).all():
        raise ValueError("Target column must contain only 0 and 1")

    df["Label"] = df["Label"].astype("int8")

    return df

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing steps to a dataset."""

    df = clean_port_columns(df)
    df = clean_categorical_columns(df)
    df = clean_attack_category(df)
    df = validate_target(df)

    return df


def clean_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """Remove surrounding whitespace from attack categories."""

    if "attack_cat" not in df.columns:
        raise ValueError("Missing column: attack_cat")

    df = df.copy()

    df["attack_cat"] = (
        df["attack_cat"]
        .astype("string")
        .str.strip()
    )

    return df
