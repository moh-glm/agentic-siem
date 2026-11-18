"""
Memory-conscious stratified sampling for Agent 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "unsw_nb15"
)

RANDOM_STATE = 42


def count_labels(files, chunksize):
    """Count normal and attack records across all files."""

    counts = {0: 0, 1: 0}

    for file_path in files:
        for chunk in pd.read_csv(
            file_path,
            usecols=["Label"],
            chunksize=chunksize,
        ):
            values = chunk["Label"].value_counts()

            for label, count in values.items():
                if label not in counts:
                    raise ValueError(f"Invalid label: {label}")

                counts[int(label)] += int(count)

    return counts


def sample_dataset(
    sample_size: int = 500_000,
    chunksize: int = 100_000,
):
    """
    Select a reproducible, stratified sample from processed files.

    Uses random priorities to maintain a uniform reservoir
    for each target class without loading the full dataset.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    files = sorted(PROCESSED_DIR.glob("*_processed.csv"))

    if not files:
        raise FileNotFoundError(
            f"No processed dataset files found in {PROCESSED_DIR}"
        )

    counts = count_labels(files, chunksize)
    total_records = sum(counts.values())

    if sample_size > total_records:
        raise ValueError(
            f"Requested {sample_size} records, "
            f"but only {total_records} are available"
        )

    # Allocate the sample proportionally between classes.
    target_counts = {
        1: round(sample_size * counts[1] / total_records),
    }
    target_counts[0] = sample_size - target_counts[1]

    rng = np.random.default_rng(RANDOM_STATE)

    features = get_feature_columns()
    columns = features + ["Label"]

    reservoirs = {
        0: pd.DataFrame(columns=columns + ["_priority"]),
        1: pd.DataFrame(columns=columns + ["_priority"]),
    }

    for file_path in files:
        print(f"Sampling from: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            usecols=columns,
            chunksize=chunksize,
        ):
            for label in (0, 1):
                target = target_counts[label]

                if target == 0:
                    continue

                class_chunk = chunk[chunk["Label"] == label].copy()

                if class_chunk.empty:
                    continue

                class_chunk["_priority"] = rng.random(
                    len(class_chunk)
                )
                if reservoirs[label].empty:
                    candidates = class_chunk
                else:
                    candidates = pd.concat(
                        [reservoirs[label], class_chunk],
                        ignore_index=True,
                    )

                reservoirs[label] = candidates.nsmallest(
                    target,
                    "_priority",
                )
    sampled = pd.concat(
        [reservoirs[0], reservoirs[1]],
        ignore_index=True,
    )

    sampled = sampled.drop(columns=["_priority"])

    sampled = sampled.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print("\nSampling completed.")
    print("Total sampled records:", len(sampled))
    print("Class distribution:")
    print(sampled["Label"].value_counts().sort_index())

    return sampled
