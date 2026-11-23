"""
Training utilities for Agent 1.
"""
import numpy as np
import pandas as pd

from src.agents.agent1.evaluation import evaluate_model
from src.data.preprocess import get_feature_columns




def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
):
    """
    Deterministic group-aware split.

    Identical feature patterns always belong to the same split.
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

    X = df[get_feature_columns()].reset_index(drop=True)
    y = df["Label"].reset_index(drop=True)

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Label must contain only 0 and 1")

    # Hash identical feature patterns to the same group.
    hashes = pd.util.hash_pandas_object(
        X,
        index=False,
    ).to_numpy(dtype=np.uint64)

    # A feature group must never have conflicting labels.
    groups = pd.DataFrame({
        "feature_hash": hashes,
        "label": y.to_numpy(),
    })

    if groups.groupby("feature_hash")["label"].nunique().gt(1).any():
        raise ValueError(
            "Identical feature groups have conflicting labels"
        )

    # Compare uint64 hashes against integer thresholds.
    hash_space = 1 << 64

    train_end = np.uint64(
        int((1 - test_size - validation_size) * hash_space)
    )
    validation_end = np.uint64(
        int((1 - test_size) * hash_space)
    )

    train_mask = hashes < train_end

    validation_mask = (
        (hashes >= train_end)
        & (hashes < validation_end)
    )

    test_mask = hashes >= validation_end

    splits = [
        ("Training", train_mask),
        ("Validation", validation_mask),
        ("Test", test_mask),
    ]

    for name, mask in splits:
        target = y.loc[mask]

        if target.empty:
            raise ValueError(f"{name} split is empty")

        if target.nunique() < 2:
            raise ValueError(
                f"{name} split must contain both classes"
            )

    return (
        X.loc[train_mask],
        X.loc[validation_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[validation_mask],
        y.loc[test_mask],
    )
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train a model and evaluate it on validation data.

    The preprocessing pipeline is fitted only on training data.
    """

    print(
        f"\nTraining "
        f"{model.named_steps['classifier'].__class__.__name__}..."
    )

    model.fit(X_train, y_train)

    print("Training completed.")
    print("Evaluating on validation data...")

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
    )

    return model, validation_results