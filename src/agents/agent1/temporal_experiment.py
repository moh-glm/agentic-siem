
"""
Chronological robustness experiment for Agent 1.

Uses the complete processed UNSW-NB15 dataset.
Records are sorted by Stime and split chronologically.
The existing baseline model and reports are never overwritten.
"""

import gc
import json
import joblib
import time
from pathlib import Path

import pandas as pd

from src.agents.agent1.models import (
    build_naive_bayes_model,
    build_logistic_regression_model,
)
from src.agents.agent1.evaluation import (
    evaluate_model,
    measure_prediction_performance,
)
from src.data.preprocess import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "unsw_nb15"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "results" / "agent1" / "temporal"
TEMPORAL_MODELS_DIR = OUTPUT_DIR / "models"

FILES = [
    "UNSW-NB15_1_processed.csv",
    "UNSW-NB15_2_processed.csv",
    "UNSW-NB15_3_processed.csv",
    "UNSW-NB15_4_processed.csv",
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def load_temporal_dataset():
    """Load required columns and sort all records by timestamp."""

    print("=" * 65)
    print("AGENT 1 — CHRONOLOGICAL DATA LOADING")
    print("=" * 65)

    columns = ["Stime", "Label"] + get_feature_columns()
    frames = []

    for filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        print(f"Loading {filename}...")

        frame = pd.read_csv(
            path,
            usecols=columns,
            low_memory=False,
        )

        frames.append(frame)

    print("\nCombining dataset...")

    df = pd.concat(frames, ignore_index=True)

    df["Stime"] = pd.to_numeric(df["Stime"], errors="coerce")

    if df["Stime"].isna().any():
        raise ValueError("Stime contains missing or invalid values.")

    if df["Label"].isna().any():
        raise ValueError("Label contains missing values.")

    if not df["Label"].isin([0, 1]).all():
        raise ValueError("Label must contain only 0 and 1.")

    df["Label"] = df["Label"].astype("int8")

    df = df.sort_values(
        "Stime",
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"Total records: {len(df):,}")
    print(f"Unique timestamps: {df['Stime'].nunique():,}")
    print(f"Time range: {df['Stime'].min()} - {df['Stime'].max()}")

    return df


def create_temporal_splits(df):
    """Split chronologically, keeping equal timestamps together."""

    unique_times = df["Stime"].drop_duplicates().to_numpy()

    train_time_index = int(len(unique_times) * TRAIN_RATIO)
    validation_time_index = int(
        len(unique_times) * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_end = unique_times[train_time_index]
    validation_end = unique_times[validation_time_index]

    train_mask = df["Stime"] < train_end

    validation_mask = (
        (df["Stime"] >= train_end)
        & (df["Stime"] < validation_end)
    )

    test_mask = df["Stime"] >= validation_end

    splits = {
        "training": df.loc[train_mask].copy(),
        "validation": df.loc[validation_mask].copy(),
        "testing": df.loc[test_mask].copy(),
    }

    print("\nChronological split:")

    for name, split in splits.items():
        if split.empty:
            raise ValueError(f"{name} split is empty.")

        if split["Label"].nunique() != 2:
            raise ValueError(
                f"{name} split must contain both classes."
            )

        print(
            f"{name.capitalize():12} | "
            f"Records: {len(split):,} | "
            f"Normal: {(split['Label'] == 0).sum():,} | "
            f"Attacks: {(split['Label'] == 1).sum():,} | "
            f"Attack rate: {split['Label'].mean() * 100:.2f}% | "
            f"Time: {split['Stime'].min()} - {split['Stime'].max()}"
        )

    return splits


def run_model_experiment(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train and evaluate a model on chronological validation data."""

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    print(f"Training time: {training_time:.2f} seconds")

    start = time.perf_counter()
    predictions = model.predict(X_validation)
    prediction_time = time.perf_counter() - start

    print(f"Prediction time: {prediction_time:.2f} seconds")

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        predictions=predictions,
    )

    throughput = measure_prediction_performance(
        model,
        X_validation,
    )

    print("\nValidation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(
        "Validation throughput: "
        f"{throughput['throughput_flows_per_second']:.2f} flows/second"
    )

    return {
        "model_name": model_name,
        "model": model,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "throughput": throughput,
        "metrics": metrics,
    }


def make_json_serializable(value):
    """Convert NumPy and pandas values into JSON-compatible objects."""

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if hasattr(value, "tolist"):
        return make_json_serializable(value.tolist())

    if hasattr(value, "item"):
        return value.item()

    return value


def run_temporal_experiment():
    """Execute chronological robustness evaluation."""

    df = load_temporal_dataset()
    splits = create_temporal_splits(df)

    feature_columns = get_feature_columns()

    X_train = splits["training"][feature_columns]
    y_train = splits["training"]["Label"]

    X_validation = splits["validation"][feature_columns]
    y_validation = splits["validation"]["Label"]

    X_test = splits["testing"][feature_columns]
    y_test = splits["testing"]["Label"]

    split_sizes = {
        "training": len(X_train),
        "validation": len(X_validation),
        "testing": len(X_test),
    }

    attack_rates = {
        "training": float(y_train.mean()),
        "validation": float(y_validation.mean()),
        "testing": float(y_test.mean()),
    }

    del df, splits
    gc.collect()

    results = []

    model_builders = [
        ("Gaussian Naive Bayes", build_naive_bayes_model),
        ("Logistic Regression", build_logistic_regression_model),
    ]

    for model_name, builder in model_builders:
        result = run_model_experiment(
            model_name,
            builder(),
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        results.append(result)
        gc.collect()

    best_model = max(
        results,
        key=lambda result: result["metrics"]["f1_score"],
    )

    print("\n" + "=" * 65)
    print("CHRONOLOGICAL MODEL COMPARISON")
    print("=" * 65)

    for result in results:
        print(
            f"{result['model_name']}: "
            f"Validation F1 = {result['metrics']['f1_score']:.4f}"
        )

    print(f"\nSelected model: {best_model['model_name']}")

    print("\n" + "=" * 65)
    print("FINAL CHRONOLOGICAL TEST EVALUATION")
    print("=" * 65)

    start = time.perf_counter()
    test_predictions = best_model["model"].predict(X_test)
    test_prediction_time = time.perf_counter() - start

    test_metrics = evaluate_model(
        best_model["model"],
        X_test,
        y_test,
        predictions=test_predictions,
    )

    test_throughput = measure_prediction_performance(
        best_model["model"],
        X_test,
    )

    for name, value in test_metrics.items():
        print(f"{name}: {value}")

    print(f"\nTest throughput: {test_throughput['throughput_flows_per_second']:.2f} flows/second")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMPORAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = TEMPORAL_MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model["model"], model_path)

    metadata = {
        "dataset": "UNSW-NB15",
        "best_model": best_model["model_name"],
        "model_file": "best_model.pkl",
        "selection_metric": "validation_f1",
        "validation_f1": float(best_model["metrics"]["f1_score"]),
        "evaluation": "chronological",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_mapping": {"0": "normal", "1": "attack"},
    }

    metadata_path = TEMPORAL_MODELS_DIR / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Temporal model saved to: {model_path}")
    print(f"Temporal metadata saved to: {metadata_path}")

    report = {
        "dataset": "UNSW-NB15",
        "evaluation": "chronological",
        "split_ratios": {
            "training": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "testing": 1 - TRAIN_RATIO - VALIDATION_RATIO,
        },
        "split_sizes": split_sizes,
        "attack_rates": attack_rates,
        "models": {
            result["model_name"]: {
                "validation_metrics": make_json_serializable(
                    result["metrics"]
                ),
                "training_time_seconds": result[
                    "training_time_seconds"
                ],
                "prediction_time_seconds": result[
                    "prediction_time_seconds"
                ],
                "throughput": make_json_serializable(
                    result["throughput"]
                ),
            }
            for result in results
        },
        "selected_model": best_model["model_name"],
        "selection_metric": "validation_f1",
        "test_metrics": make_json_serializable(test_metrics),
        "test_prediction_time_seconds": test_prediction_time,
        "test_throughput": make_json_serializable(test_throughput),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_mapping": {
            "0": "normal",
            "1": "attack",
        },
    }

    report_path = OUTPUT_DIR / "temporal_results.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nTemporal report saved to: {report_path}")
    print("\nTemporal experiment completed successfully.")

    return report


if __name__ == "__main__":
    run_temporal_experiment()
