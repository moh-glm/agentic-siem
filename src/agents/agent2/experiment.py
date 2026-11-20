"""
Real-data experiments for Agent 2 (RNN & LSTM Sequence Analysis).

Trains and evaluates SequenceRNN and SequenceLSTM on chronological
sliding-window sequences extracted from UNSW-NB15.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.agents.agent1.evaluation import evaluate_model
from src.agents.agent1.preprocessing import build_preprocessor
from src.agents.agent1.sampling import sample_dataset
from src.agents.agent2.models import build_rnn_model
from src.agents.agent2.sequence_builder import create_sequences, sort_by_time


# Experiment configuration matching Agent 1 scale
SAMPLE_SIZE = 500_000
CHUNKSIZE = 100_000
SEQUENCE_LENGTH = 10
STRIDE = 5
BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "agent2"
MODELS_DIR = PROJECT_ROOT / "models"


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sequence_data(
    sample_size: int = SAMPLE_SIZE,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
):
    """
    Load UNSW-NB15 sample, sort chronologically, preprocess features,
    and build train/validation/test sequence datasets.
    """
    print("=" * 60)
    print("AGENT 2 — CHRONOLOGICAL SEQUENCE DATA PREPARATION")
    print("=" * 60)

    start_time = time.perf_counter()
    df = sample_dataset(sample_size=sample_size, chunksize=CHUNKSIZE)
    df = sort_by_time(df, time_col="Stime")

    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)

    preprocessor = build_preprocessor(scale_numerical=True, dense_output=True)
    X_train_2d = preprocessor.fit_transform(df_train)
    X_val_2d = preprocessor.transform(df_val)
    X_test_2d = preprocessor.transform(df_test)

    y_train_1d = df_train["Label"].to_numpy()
    y_val_1d = df_val["Label"].to_numpy()
    y_test_1d = df_test["Label"].to_numpy()

    X_train_seq, y_train_seq = create_sequences(
        X_train_2d, y_train_1d, sequence_length=sequence_length, stride=stride
    )
    X_val_seq, y_val_seq = create_sequences(
        X_val_2d, y_val_1d, sequence_length=sequence_length, stride=stride
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_2d, y_test_1d, sequence_length=sequence_length, stride=stride
    )

    elapsed = time.perf_counter() - start_time
    print(f"Train sequences:      {len(X_train_seq):,} | Shape: {X_train_seq.shape}")
    print(f"Validation sequences: {len(X_val_seq):,} | Shape: {X_val_seq.shape}")
    print(f"Testing sequences:    {len(X_test_seq):,} | Shape: {X_test_seq.shape}")
    print(f"Preparation time:     {elapsed:.2f} seconds")

    return (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        X_test_seq,
        y_test_seq,
    )


def predict_sequences(model: nn.Module, X_seq: np.ndarray, device: torch.device) -> np.ndarray:
    """Run batched inference on sequence data and return binary predictions."""
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X_seq))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    preds = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            probs = torch.sigmoid(model(batch_x))
            preds.append((probs >= 0.5).long().cpu().numpy())

    return np.concatenate(preds, axis=0)


def train_and_evaluate_sequence_model(
    model_name: str,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = EPOCHS,
) -> dict:
    """Train one PyTorch sequence model and evaluate on validation sequences."""
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name} (device={device})")
    print("=" * 60)

    input_size = X_train.shape[2]
    model = build_rnn_model(model_type=model_type, input_size=input_size).to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_start = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    training_time = time.perf_counter() - training_start

    prediction_start = time.perf_counter()
    val_predictions = predict_sequences(model, X_val, device)
    prediction_time = time.perf_counter() - prediction_start

    metrics = evaluate_model(model=None, X_test=None, y_test=y_val, predictions=val_predictions)
    print("\nValidation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "metrics": metrics,
        "model": model,
    }


def run_experiments(
    sample_size: int = SAMPLE_SIZE,
    epochs: int = EPOCHS,
):
    """Run Vanilla RNN and LSTM sequence experiments and save results."""
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = prepare_sequence_data(sample_size=sample_size)

    results = []
    for model_name, model_type in [
        ("Vanilla RNN", "rnn"),
        ("LSTM", "lstm"),
    ]:
        res = train_and_evaluate_sequence_model(
            model_name=model_name,
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            epochs=epochs,
        )
        results.append(res)

    best_model = max(results, key=lambda r: r["metrics"]["f1_score"])
    print(f"\nBest model based on validation F1-score: {best_model['model_name']}")

    test_predictions = predict_sequences(best_model["model"], X_test, device)
    test_metrics = evaluate_model(
        model=None, X_test=None, y_test=y_test, predictions=test_predictions
    )

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    model_path = MODELS_DIR / "agent2_best_model.pt"
    torch.save(
        {
            "model_type": best_model["model_type"],
            "input_size": X_train.shape[2],
            "sequence_length": SEQUENCE_LENGTH,
            "state_dict": best_model["model"].state_dict(),
        },
        model_path,
    )
    print(f"Best model weights saved to: {model_path}")

    report = {
        "dataset": "UNSW-NB15",
        "sample_size": sample_size,
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "split": {
            "training_sequences": len(X_train),
            "validation_sequences": len(X_val),
            "testing_sequences": len(X_test),
        },
        "models": [
            {
                "model_name": r["model_name"],
                "training_time_seconds": r["training_time_seconds"],
                "prediction_time_seconds": r["prediction_time_seconds"],
                "validation_metrics": r["metrics"],
            }
            for r in results
        ],
        "best_model": best_model["model_name"],
        "test_metrics": test_metrics,
    }

    output_file = RESULTS_DIR / "experiment_results.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Results saved to: {output_file}")
    return results, best_model


if __name__ == "__main__":
    run_experiments()
