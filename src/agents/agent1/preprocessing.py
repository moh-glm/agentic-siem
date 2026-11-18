
"""
Preprocessing pipelines for Agent 1.

Provides separate preprocessing configurations
for Gaussian Naive Bayes and Logistic Regression.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.data.preprocess import (
    CATEGORICAL_COLUMNS,
    get_feature_columns,
)


def get_numerical_columns():
    """Return numerical features."""

    return [
        column
        for column in get_feature_columns()
        if column not in CATEGORICAL_COLUMNS
    ]


def clean_numerical_values(X):
    """Convert whitespace-only numerical values to NaN."""

    import numpy as np
    import pandas as pd

    X = pd.DataFrame(X).copy()
    X = X.replace(r"^\s*$", np.nan, regex=True)

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    return X.to_numpy(dtype=float)


def build_preprocessor(
    scale_numerical: bool = True,
    dense_output: bool = False,
):
    """Build numerical and categorical preprocessing pipelines."""

    numerical_steps = [
        (
            "cleaner",
            FunctionTransformer(
                clean_numerical_values,
                feature_names_out="one-to-one",
            ),
        ),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numerical:
        numerical_steps.append(("scaler", StandardScaler()))

    numerical_pipeline = Pipeline(numerical_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="unknown",
            ),
        ),
        (
            "encoder",
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=not dense_output,
),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                get_numerical_columns(),
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
sparse_threshold=0 if dense_output else 0.3,
    )


