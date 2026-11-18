"""
Inspect the structure of the UNSW-NB15 dataset.
"""

import pandas as pd

from src.data.dataset_loader import DATA_DIR, FILES, COLUMNS


def inspect_dataset():
    """Display a sample of each raw dataset file."""

    for filename in FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        print("\n" + "=" * 60)
        print(f"FILE: {filename}")
        print("=" * 60)

        df = pd.read_csv(
            file_path,
            header=None,
            names=COLUMNS,
            nrows=5,
        )

        print("Sample shape:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 records:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    inspect_dataset()
