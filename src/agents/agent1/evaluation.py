"""
Evaluation utilities for Agent 1.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, predictions=None):
    """
    Evaluate a trained binary classification model.

    Class 1 represents an attack.
    """

    if predictions is None:
        predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    return results
