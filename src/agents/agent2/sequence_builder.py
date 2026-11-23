"""
Sequence construction utilities for Agent 2 (RNN Behaviour Analysis).

Converts tabular network flow records into ordered temporal sequences
using a sliding window approach while strictly respecting file/session
boundaries (never building a window across two separate CSV files).

Sequence-Labeling Rule:
    A sequence window of length L is classified as an attack (label = 1)
    if at least one flow inside that window has Label == 1; otherwise it is
    classified as normal (label = 0).
"""

from typing import Sequence
import numpy as np
import pandas as pd


def sort_by_time(df: pd.DataFrame, time_col: str = "Stime") -> pd.DataFrame:
    """
    Sort network flows chronologically within a single file or session.
    """
    if time_col in df.columns:
        df = df.copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.sort_values(by=time_col, kind="mergesort").reset_index(drop=True)
    return df


def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    """Validate that labels are 1D, finite, and strictly binary (0 or 1)."""
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"Expected 1D label array, got shape {y_arr.shape}.")
    if np.issubdtype(y_arr.dtype, np.floating) and not np.all(np.isfinite(y_arr)):
        raise ValueError("Sequence labels contain NaN or Inf values.")
    unique_vals = set(np.unique(y_arr).tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Labels must be binary (0 or 1), got values: {sorted(unique_vals)}")
    return y_arr.astype(np.int64)


def create_sequences(
    X: np.ndarray,
    y: np.ndarray | None = None,
    sequence_length: int = 10,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build sliding-window sequences from a single contiguous 2D feature array.

    Sequence-Labeling Rule:
        A sequence is labeled as an attack (1) if at least one flow in its
        window is an attack (y == 1), and normal (0) otherwise.

    Args:
        X: 2D array of shape (num_samples, num_features) from a single file/session.
        y: Optional 1D binary label array of shape (num_samples,).
        sequence_length: Number of consecutive flows per sequence (>= 2).
        stride: Step size between consecutive windows (>= 1).

    Returns:
        X_seq: 3D array of shape (num_sequences, sequence_length, num_features).
        y_seq: 1D array of shape (num_sequences,) or None if y is not provided.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2 for temporal analysis.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Input features X contain NaN or Inf values.")

    num_samples = len(X_arr)
    if num_samples < sequence_length:
        raise ValueError(
            f"Not enough samples ({num_samples}) to build a sequence of length {sequence_length}."
        )

    y_arr = None
    if y is not None:
        y_arr = _validate_binary_labels(y)
        if len(y_arr) != num_samples:
            raise ValueError(
                f"Length mismatch: X has {num_samples} samples, y has {len(y_arr)} labels."
            )

    sequences = []
    labels = []

    for start in range(0, num_samples - sequence_length + 1, stride):
        end = start + sequence_length
        sequences.append(X_arr[start:end])
        if y_arr is not None:
            # Sequence-labeling rule: attack (1) if at least one flow in window is attack
            labels.append(int(np.any(y_arr[start:end] == 1)))

    X_seq = np.stack(sequences, axis=0)
    y_seq = np.array(labels, dtype=np.int64) if y_arr is not None else None

    return X_seq, y_seq


def create_sequences_per_file(
    X_list: Sequence[np.ndarray],
    y_list: Sequence[np.ndarray] | None = None,
    sequence_length: int = 10,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Generate sliding-window sequences independently per file/session, then
    concatenate the resulting sequences so no window ever crosses a file boundary.
    """
    if not X_list:
        raise ValueError("X_list cannot be empty.")
    if y_list is not None and len(X_list) != len(y_list):
        raise ValueError(
            f"Mismatch between number of feature arrays ({len(X_list)}) and label arrays ({len(y_list)})."
        )

    all_X_seq = []
    all_y_seq = []

    for idx, X_part in enumerate(X_list):
        y_part = y_list[idx] if y_list is not None else None
        X_seq, y_seq = create_sequences(
            X=X_part,
            y=y_part,
            sequence_length=sequence_length,
            stride=stride,
        )
        all_X_seq.append(X_seq)
        if y_seq is not None:
            all_y_seq.append(y_seq)

    X_concat = np.concatenate(all_X_seq, axis=0)
    y_concat = np.concatenate(all_y_seq, axis=0) if y_list is not None else None
    return X_concat, y_concat
