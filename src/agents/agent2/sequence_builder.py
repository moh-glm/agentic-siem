"""
Sequence construction utilities for Agent 2 (RNN Behaviour Analysis).

Converts tabular network flow records into ordered temporal sequences
using a sliding window approach.
"""

import numpy as np
import pandas as pd


def sort_by_time(df: pd.DataFrame, time_col: str = "Stime") -> pd.DataFrame:
    """
    Sort network flows chronologically if the timestamp column exists.
    """
    if time_col in df.columns:
        df = df.copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.sort_values(by=time_col, kind="mergesort").reset_index(drop=True)
    return df


def create_sequences(
    X: np.ndarray,
    y: np.ndarray | None = None,
    sequence_length: int = 10,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build sliding-window sequences from 2D preprocessed feature arrays.

    Args:
        X: 2D array of shape (num_samples, num_features).
        y: Optional 1D binary label array of shape (num_samples,).
        sequence_length: Number of consecutive flows per sequence.
        stride: Step size between consecutive windows.

    Returns:
        X_seq: 3D array of shape (num_sequences, sequence_length, num_features).
        y_seq: 1D array of shape (num_sequences,) where 1 indicates at least
               one attack flow in the window, or None if y is not provided.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2 for temporal analysis.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {X_arr.shape}")

    num_samples = len(X_arr)
    if num_samples < sequence_length:
        raise ValueError(
            f"Not enough samples ({num_samples}) to build a sequence of length {sequence_length}."
        )

    sequences = []
    labels = []
    y_arr = np.asarray(y, dtype=np.int64) if y is not None else None

    for start in range(0, num_samples - sequence_length + 1, stride):
        end = start + sequence_length
        sequences.append(X_arr[start:end])
        if y_arr is not None:
            # A sequence is marked suspicious (1) if any flow inside it is an attack
            labels.append(int(np.any(y_arr[start:end] == 1)))

    X_seq = np.stack(sequences, axis=0)
    y_seq = np.array(labels, dtype=np.int64) if y_arr is not None else None

    return X_seq, y_seq
