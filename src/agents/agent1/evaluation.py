
"""
Evaluation utilities for Agent 1.

Class 0: Normal traffic.
Class 1: Attack traffic.
"""

import time

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

    Includes classification metrics, cybersecurity error rates,
    operational error counts, throughput, and ranking metrics.
    """

    y_test = np.asarray(y_test)

    if predictions is None:
        predictions = model.predict(X_test)

    predictions = np.asarray(predictions)

    if len(y_test) == 0:
        raise ValueError("Evaluation dataset cannot be empty.")

    if len(predictions) != len(y_test):
        raise ValueError("Predictions and labels must have equal lengths.")

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Predictions must contain only 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    normal_count = tn + fp
    attack_count = fn + tp

    specificity = tn / normal_count if normal_count else 0.0
    false_positive_rate = fp / normal_count if normal_count else 0.0
    false_negative_rate = fn / attack_count if attack_count else 0.0

    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_test, predictions))
            if len(np.unique(y_test)) == 2
            else None
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "normal_flows": int(normal_count),
        "attack_flows": int(attack_count),
        "false_positives_per_1000_normal_flows": (
            float(fp / normal_count * 1000) if normal_count else 0.0
        ),
        "false_negatives_per_1000_attack_flows": (
            float(fn / attack_count * 1000) if attack_count else 0.0
        ),
    }

    if hasattr(model, "predict_proba"):
        probability_matrix = np.asarray(model.predict_proba(X_test))
        classes = np.asarray(getattr(model, "classes_", [0, 1]))

        positive_indices = np.flatnonzero(classes == 1)

        if (
            probability_matrix.ndim != 2
            or probability_matrix.shape[0] != len(y_test)
            or probability_matrix.shape[1] != len(classes)
            or len(positive_indices) != 1
        ):
            raise ValueError(
                "Invalid probability matrix or missing attack class (1)."
            )

        probabilities = probability_matrix[:, positive_indices[0]]

        if not np.isfinite(probabilities).all():
            raise ValueError("Predicted probabilities contain invalid values.")

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Predicted probabilities must be between 0 and 1.")

        if len(np.unique(y_test)) == 2:
            results["roc_auc"] = float(
                roc_auc_score(y_test, probabilities)
            )
            results["average_precision"] = float(
                average_precision_score(y_test, probabilities)
            )
        else:
            results["roc_auc"] = None
            results["average_precision"] = None

    return results


def measure_prediction_performance(model, X_test):
    """
    Measure prediction latency and throughput.

    This is a separate benchmark, not part of classification evaluation.
    """

    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start

    total_records = len(X_test)

    return {
        "records": int(total_records),
        "prediction_time_seconds": float(elapsed),
        "throughput_flows_per_second": (
            float(total_records / elapsed) if elapsed > 0 else None
        ),
    }
