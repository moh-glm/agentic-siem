"""
Training utilities for Agent 1.
"""

import pandas as pd
from src.agents.agent1.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

from src.data.preprocess import get_feature_columns


RANDOM_STATE = 42


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
):
    """
    Split the dataset into training, validation and test sets.

    Uses stratification to preserve the normal/attack ratio.
    """

    if "Label" not in df.columns:
        raise ValueError("Dataset must contain the Label column")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1"
        )

    X = df[get_feature_columns()]
    y = df["Label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    relative_test_size = (
        test_size / (test_size + validation_size)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train a model and evaluate it on validation data.

    The preprocessing pipeline is fitted only on training data.
    """

    print(f"\nTraining {model.named_steps['classifier'].__class__.__name__}...")

    model.fit(X_train, y_train)

    print("Training completed.")

    print("Evaluating on validation data...")

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
    )

    return model, validation_results
