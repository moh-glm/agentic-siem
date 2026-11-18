"""
Real-data experiments for Agent 1.

Compares Gaussian Naive Bayes and Logistic Regression
on a stratified sample of UNSW-NB15.
"""

import time
import json
import resource
from pathlib import Path

import pandas as pd

from src.agents.agent1.sampling import sample_dataset
from src.agents.agent1.train import split_dataset, train_model
from src.agents.agent1.models import (
    build_naive_bayes_model,
    build_logistic_regression_model,
)


# Experiment configuration

SAMPLE_SIZE = 500_000
CHUNKSIZE = 100_000
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "agent1"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
def prepare_experiment_data():
    """
    Sample UNSW-NB15 and split it into training,
    validation and testing sets.
    """

    print("=" * 60)
    print("AGENT 1 — DATA PREPARATION")
    print("=" * 60)

    start_time = time.perf_counter()

    df = sample_dataset(
        sample_size=SAMPLE_SIZE,
        chunksize=CHUNKSIZE,
    )

    print("\nSplitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_dataset(
            df,
            test_size=0.15,
            validation_size=0.15,
        )
    )

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

    return X_train, X_val, X_test, y_train, y_val, y_test
def run_model_experiment(model_name, model, X_train, y_train, X_val, y_val):
    """
    Train and evaluate one model on the validation set.
    """

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    # Measure training time
    training_start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - training_start

    print(f"Training time: {training_time:.2f} seconds")

    # Measure prediction time
    prediction_start = time.perf_counter()

    predictions = model.predict(X_val)

    prediction_time = time.perf_counter() - prediction_start

    print(f"Prediction time: {prediction_time:.2f} seconds")

    # Evaluate the model
    from src.agents.agent1.evaluation import evaluate_model

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
    Run Gaussian Naive Bayes and Logistic Regression experiments.
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

    # Experiment 1: Gaussian Naive Bayes
    naive_bayes = build_naive_bayes_model()

    nb_results = run_model_experiment(
        "Gaussian Naive Bayes",
        naive_bayes,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    results.append(nb_results)

    # Experiment 2: Logistic Regression
    logistic_regression = build_logistic_regression_model()

    lr_results = run_model_experiment(
        "Logistic Regression",
        logistic_regression,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    results.append(lr_results)

    # Compare validation F1-scores
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

    print("\nBest model based on validation F1-score:")
    print(best_model["model_name"])
        # Final evaluation on the untouched test set
    from src.agents.agent1.evaluation import evaluate_model

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

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # Save complete experiment results
    report = {
        "dataset": "UNSW-NB15",
        "sample_size": SAMPLE_SIZE,
        "split": {
            "training": len(X_train),
            "validation": len(X_val),
            "testing": len(X_test),
        },
        "models": [
            {
                "model_name": result["model_name"],
                "training_time_seconds": result["training_time_seconds"],
                "prediction_time_seconds": result["prediction_time_seconds"],
                "validation_metrics": result["metrics"],
            }
            for result in results
        ],
        "best_model": best_model["model_name"],
        "test_metrics": test_metrics,
    }

    output_file = RESULTS_DIR / "experiment_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nResults saved to: {output_file}")
    return results, best_model, (X_test, y_test)


if __name__ == "__main__":
    run_experiments()
