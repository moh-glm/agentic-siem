
"""
Real-data experiments for Agent 1.

Compares Gaussian Naive Bayes and Logistic Regression
on a stratified sample of UNSW-NB15.

The best model is selected using validation F1-score.
The untouched test set is used only for final evaluation.
"""

import json
import time
from pathlib import Path

import joblib

from src.agents.agent1.sampling import sample_dataset
from src.agents.agent1.train import split_dataset
from src.agents.agent1.models import (
    build_naive_bayes_model,
    build_logistic_regression_model,
)
from src.agents.agent1.evaluation import evaluate_model
from src.data.preprocess import get_feature_columns


# Experiment configuration

SAMPLE_SIZE = 500_000
CHUNKSIZE = 100_000

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "agent1"
MODELS_DIR = RESULTS_DIR / "models"

MODEL_FILE = "best_model.pkl"
METADATA_FILE = "metadata.json"
REPORT_FILE = "experiment_results.json"

TARGET_MAPPING = {
    "0": "normal",
    "1": "attack",
}


def prepare_experiment_data():
    """Sample UNSW-NB15 and create train, validation and test sets."""

    print("=" * 60)
    print("AGENT 1 — DATA PREPARATION")
    print("=" * 60)

    start_time = time.perf_counter()

    df = sample_dataset(
        sample_size=SAMPLE_SIZE,
        chunksize=CHUNKSIZE,
    )

    if df.empty:
        raise ValueError("The sampled dataset is empty.")

    print("\nSplitting dataset...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        df,
        test_size=0.15,
        validation_size=0.15,
    )

    if any(
        len(split) == 0
        for split in (X_train, X_val, X_test)
    ):
        raise ValueError("One or more dataset splits are empty.")

    print("\nDataset split completed.")

    print(f"Training records:   {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Testing records:    {len(X_test):,}")

    print("\nClass distribution:")

    for name, labels in [
        ("Training", y_train),
        ("Validation", y_val),
        ("Testing", y_test),
    ]:
        attack_percentage = labels.mean() * 100

        print(
            f"{name:12} | "
            f"Normal: {(labels == 0).sum():,} | "
            f"Attacks: {(labels == 1).sum():,} | "
            f"Attack rate: {attack_percentage:.2f}%"
        )

    elapsed = time.perf_counter() - start_time

    print(f"\nPreparation time: {elapsed:.2f} seconds")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def run_model_experiment(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """Train one model and evaluate it on the validation set."""

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    training_start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - training_start

    print(f"Training time: {training_time:.2f} seconds")

    prediction_start = time.perf_counter()

    predictions = model.predict(X_val)

    prediction_time = time.perf_counter() - prediction_start

    print(f"Prediction time: {prediction_time:.2f} seconds")

    metrics = evaluate_model(
        model,
        X_val,
        y_val,
        predictions=predictions,
    )

    print("\nValidation metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return {
        "model_name": model_name,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "metrics": metrics,
        "model": model,
    }


def run_experiments():
    """
    Train both models, select the best using validation F1,
    evaluate it on the untouched test set, and save its artifact.
    """

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = prepare_experiment_data()

    results = []

    models = [
        (
            "Gaussian Naive Bayes",
            build_naive_bayes_model(),
        ),
        (
            "Logistic Regression",
            build_logistic_regression_model(),
        ),
    ]

    for model_name, model in models:
        result = run_model_experiment(
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        results.append(result)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for result in results:
        print(
            f"{result['model_name']}: "
            f"F1 = {result['metrics']['f1_score']:.4f}"
        )

    best_model = max(
        results,
        key=lambda result: result["metrics"]["f1_score"],
    )

    best_validation_f1 = float(
        best_model["metrics"]["f1_score"]
    )

    print("\nBest model based on validation F1-score:")
    print(best_model["model_name"])

    # Final evaluation on the untouched test set.
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_predictions = best_model["model"].predict(X_test)

    test_metrics = evaluate_model(
        best_model["model"],
        X_test,
        y_test,
        predictions=test_predictions,
    )

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # Prepare output directory.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save only the selected model.
    model_path = MODELS_DIR / MODEL_FILE

    joblib.dump(
        best_model["model"],
        model_path,
    )

    print(f"\nBest model saved to: {model_path}")

    # Save metadata required by Agent1 runtime.
    metadata = {
        "dataset": "UNSW-NB15",
        "best_model": best_model["model_name"],
        "model_file": MODEL_FILE,
        "selection_metric": "validation_f1",
        "validation_f1": best_validation_f1,
        "test_metrics": {
            metric: float(value)
            for metric, value in test_metrics.items()
        },
        "sample_size": len(X_train) + len(X_val) + len(X_test),
        "split_sizes": {
            "training": len(X_train),
            "validation": len(X_val),
            "testing": len(X_test),
        },
        "models": {
            result["model_name"]: {
                "validation_f1": float(
                    result["metrics"]["f1_score"]
                )
            }
            for result in results
        },
        "feature_count": len(get_feature_columns()),
        "feature_columns": get_feature_columns(),
        "target_mapping": TARGET_MAPPING,
    }

    metadata_file = MODELS_DIR / METADATA_FILE

    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Model metadata saved to: {metadata_file}")

    # Save complete experiment results.
    report = {
        "dataset": "UNSW-NB15",
        "sample_size": len(X_train) + len(X_val) + len(X_test),
        "split": {
            "training": len(X_train),
            "validation": len(X_val),
            "testing": len(X_test),
        },
        "models": [
            {
                "model_name": result["model_name"],
                "training_time_seconds": result[
                    "training_time_seconds"
                ],
                "prediction_time_seconds": result[
                    "prediction_time_seconds"
                ],
                "validation_metrics": result["metrics"],
            }
            for result in results
        ],
        "best_model": best_model["model_name"],
        "selection_metric": "validation_f1",
        "test_metrics": test_metrics,
    }

    output_file = RESULTS_DIR / REPORT_FILE

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nResults saved to: {output_file}")

    return results, best_model, (X_test, y_test)


if __name__ == "__main__":
    run_experiments()
