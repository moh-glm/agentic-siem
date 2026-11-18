"""
Preprocessing pipelines for Agent 1.

Provides separate preprocessing configurations
for Gaussian Naive Bayes and Logistic Regression.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def build_preprocessor(scale_numerical: bool = True):
    """
    Build a preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Optional standard scaling

    Categorical features:
        - Most frequent value imputation
        - One-hot encoding
    """

    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numerical:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

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
                sparse_output=True,
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
    )
