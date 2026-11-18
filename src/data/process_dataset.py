
"""
Process the complete UNSW-NB15 dataset in chunks.

Each source CSV is cleaned and saved separately.
"""

from pathlib import Path

import pandas as pd

from src.data.dataset_loader import DATA_DIR, FILES, COLUMNS
from src.data.preprocess import preprocess_dataset


# Process 100,000 records at a time.
CHUNK_SIZE = 100_000

# Output directory.
OUTPUT_DIR = DATA_DIR.parent.parent / "processed" / "unsw_nb15"


def process_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_normal = 0
    total_attacks = 0

    # Prevent accidental overwriting of existing results.
    for filename in FILES:
        output_path = OUTPUT_DIR / f"{Path(filename).stem}_processed.csv"
        temp_path = OUTPUT_DIR / f"{Path(filename).stem}_processed.csv.tmp"

        if output_path.exists() or temp_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Move or rename existing output files before retrying."
            )

    for filename in FILES:
        input_path = DATA_DIR / filename
        output_path = OUTPUT_DIR / f"{Path(filename).stem}_processed.csv"
        temp_path = OUTPUT_DIR / f"{Path(filename).stem}_processed.csv.tmp"

        print(f"\nProcessing: {filename}")

        file_records = 0
        first_chunk = True

        try:
            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    input_path,
                    header=None,
                    names=COLUMNS,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                ),
                start=1,
            ):
                processed = preprocess_dataset(chunk)

                # Update statistics.
                file_records += len(processed)
                total_normal += int((processed["Label"] == 0).sum())
                total_attacks += int((processed["Label"] == 1).sum())

                # Save the processed chunk.
                processed.to_csv(
                    temp_path,
                    index=False,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                )

                first_chunk = False

                print(
                    f"  Chunk {chunk_number}: "
                    f"{len(processed):,} records processed"
                )

            # Publish the file only after successful processing.
            if first_chunk:
                raise ValueError(f"No records found in {filename}")

            temp_path.replace(output_path)

            total_records += file_records

            print(f"Completed: {file_records:,} records")
            print(f"Saved to: {output_path}")

        except Exception:
            print(f"Processing failed for {filename}.")
            print(f"Temporary output, if present: {temp_path}")
            raise

    print("\n" + "=" * 45)
    print("DATASET PROCESSING COMPLETED")
    print("=" * 45)
    print(f"Total records: {total_records:,}")
    print(f"Normal records: {total_normal:,}")
    print(f"Attack records: {total_attacks:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_dataset()
