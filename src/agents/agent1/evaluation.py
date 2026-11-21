"""
Evaluation utilities for Agent 1.

Class 0: Normal traffic.
Class 1: Attack traffic.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, predictions=None):
    """
    Evaluate a binary classification model.

    Returns standard classification metrics, cybersecurity-specific
    error rates, and probability-based ranking metrics when available.
    """

    y_test = np.asarray(y_test)

    if predictions is None:
        predictions = model.predict(X_test)

    predictions = np.asarray(predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

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
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": (
            balanced_accuracy_score(y_test, predictions)
            if len(np.unique(y_test)) == 2
            else None
        ),        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if hasattr(model, "predict_proba"):
        probability_matrix = np.asarray(model.predict_proba(X_test))
        classes = np.asarray(
            getattr(model, "classes_", [0, 1])
        )

        positive_indices = np.flatnonzero(classes == 1)

        if (
            probability_matrix.ndim != 2
            or probability_matrix.shape[1] != len(classes)
            or len(positive_indices) != 1
        ):
            raise ValueError(
                "Invalid probability matrix or missing attack class (1)."
            )

        probabilities = probability_matrix[:, positive_indices[0]]

        if len(np.unique(y_test)) == 2:
            results["roc_auc"] = roc_auc_score(
                y_test, probabilities
            )
            results["average_precision"] = average_precision_score(
                y_test, probabilities
            )
        else:
            results["roc_auc"] = None
            results["average_precision"] = None

    return results
