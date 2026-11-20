"""
Real-data experiments for Agent 2 (RNN & LSTM Sequence Analysis).

Trains and evaluates SequenceRNN and SequenceLSTM on chronological
sliding-window sequences built independently per UNSW-NB15 CSV file
(preventing cross-file sliding windows).

Sequence-Labeling Rule:
    A sequence window is labeled as an attack (1) if at least one flow inside
    the window has Label == 1; otherwise normal (0).
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.agents.agent1.evaluation import evaluate_model
from src.agents.agent1.preprocessing import build_preprocessor
from src.agents.agent2.models import build_rnn_model
from src.agents.agent2.sequence_builder import create_sequences_per_file, sort_by_time
from src.data.dataset_loader import COLUMNS, DATA_DIR, FILES
from src.data.preprocess import preprocess_dataset


PER_FILE_SAMPLE = 125_000  # 125,000 x 4 files = 500,000 records
SEQUENCE_LENGTH = 10
STRIDE = 5
BATCH_SIZE = 512
EPOCHS = 12
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
    raw_dir: Path = DATA_DIR,
    per_file_sample: int = PER_FILE_SAMPLE,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
):
    """
    Load contiguous chronological blocks per UNSW-NB15 file, split temporally
    (70/15/15) within each file, preprocess features, and generate sequences
    independently per file before concatenating.
    """
    print("=" * 60)
    print("AGENT 2 — PER-FILE CHRONOLOGICAL SEQUENCE PREPARATION")
    print("=" * 60)

    start_time = time.perf_counter()
    train_dfs, val_dfs, test_dfs = [], [], []

    for fname in FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Dataset file not found: {fpath}")

        print(f"Loading and sorting chronological block from {fname}...")
        df_part = pd.read_csv(
            fpath, header=None, names=COLUMNS, nrows=per_file_sample, low_memory=False
        )
        df_part = preprocess_dataset(df_part)
        df_part = sort_by_time(df_part, time_col="Stime")

        n = len(df_part)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)
        train_dfs.append(df_part.iloc[:t_end].reset_index(drop=True))
        val_dfs.append(df_part.iloc[t_end:v_end].reset_index(drop=True))
        test_dfs.append(df_part.iloc[v_end:].reset_index(drop=True))

    # Fit preprocessor on combined training data without mixing file order
    preprocessor = build_preprocessor(scale_numerical=True, dense_output=True)
    preprocessor.fit(pd.concat(train_dfs, ignore_index=True))

    X_train_list = [preprocessor.transform(df) for df in train_dfs]
    y_train_list = [df["Label"].to_numpy() for df in train_dfs]

    X_val_list = [preprocessor.transform(df) for df in val_dfs]
    y_val_list = [df["Label"].to_numpy() for df in val_dfs]

    X_test_list = [preprocessor.transform(df) for df in test_dfs]
    y_test_list = [df["Label"].to_numpy() for df in test_dfs]

    # Build sliding windows independently per file, then concatenate
    X_train_seq, y_train_seq = create_sequences_per_file(
        X_train_list, y_train_list, sequence_length=sequence_length, stride=stride
    )
    X_val_seq, y_val_seq = create_sequences_per_file(
        X_val_list, y_val_list, sequence_length=sequence_length, stride=stride
    )
    X_test_seq, y_test_seq = create_sequences_per_file(
        X_test_list, y_test_list, sequence_length=sequence_length, stride=stride
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


def evaluate_epoch_validation(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Compute validation loss, validation F1-score, and binary predictions."""
    model.eval()
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).float(),
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    total_loss = 0.0
    preds = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_x)
            probs = torch.sigmoid(logits)
            preds.append((probs >= 0.5).long().cpu().numpy())

    avg_val_loss = total_loss / len(val_dataset)
    val_preds = np.concatenate(preds, axis=0)
    val_f1 = float(f1_score(y_val, val_preds, zero_division=0))
    return avg_val_loss, val_f1, val_preds


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
    """
    Train one PyTorch sequence model, track validation loss and validation F1
    after every epoch, and restore the best checkpoint based on validation F1.
    """
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

    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epoch_history = []

    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        avg_train_loss = epoch_loss / len(train_dataset)
        val_loss, val_f1, _ = evaluate_epoch_validation(
            model, X_val, y_val, criterion, device
        )

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_f1": round(val_f1, 6),
            }
        )
        marker = " * (best)" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs} - Train Loss: {avg_train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}{marker}"
        )

    training_time = time.perf_counter() - training_start

    # Restore best checkpoint based on validation F1
    model.load_state_dict(best_state_dict)
    print(f"Restored best {model_name} checkpoint from epoch {best_epoch} (Val F1={best_val_f1:.4f})")

    prediction_start = time.perf_counter()
    val_predictions = predict_sequences(model, X_val, device)
    prediction_time = time.perf_counter() - prediction_start

    metrics = evaluate_model(model=None, X_test=None, y_test=y_val, predictions=val_predictions)
    print("\nBest Checkpoint Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "best_epoch": best_epoch,
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "epoch_history": epoch_history,
        "metrics": metrics,
        "model": model,
    }


def run_experiments(
    raw_dir: Path = DATA_DIR,
    per_file_sample: int = PER_FILE_SAMPLE,
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
    ) = prepare_sequence_data(raw_dir=raw_dir, per_file_sample=per_file_sample)

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
            "input_size": int(X_train.shape[2]),
            "hidden_size": int(best_model["model"].hidden_size),
            "num_layers": int(best_model["model"].num_layers),
            "dropout": float(best_model["model"].dropout),
            "sequence_length": SEQUENCE_LENGTH,
            "threshold": 0.5,
            "best_epoch": int(best_model["best_epoch"]),
            "state_dict": best_model["model"].state_dict(),
        },
        model_path,
    )
    print(f"Best model weights saved to: {model_path}")

    report = {
        "dataset": "UNSW-NB15",
        "sample_size": per_file_sample * len(FILES),
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "sequence_boundary_policy": "per_file_independent_no_cross_file_windows",
        "sequence_labeling_rule": "attack (1) if at least one flow in window is an attack, else normal (0)",
        "split": {
            "training_sequences": int(len(X_train)),
            "validation_sequences": int(len(X_val)),
            "testing_sequences": int(len(X_test)),
        },
        "models": [
            {
                "model_name": r["model_name"],
                "best_epoch": r["best_epoch"],
                "training_time_seconds": r["training_time_seconds"],
                "prediction_time_seconds": r["prediction_time_seconds"],
                "epoch_history": r["epoch_history"],
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
