# UNSW-NB15 Dataset Analysis

## Dataset Overview

- Dataset: UNSW-NB15
- Total records: 2,540,047
- Total columns: 49
- CSV files: 4
- Normal records: 2,218,764
- Attack records: 321,283

## Class Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Normal (0) | 2,218,764 | 87.35% |
| Attack (1) | 321,283 | 12.65% |

## Data Quality

- All four CSV files were successfully read.
- All files contain 49 columns.
- No inconsistencies were found between binary labels
  and attack-category missingness.
- Attack categories contain inconsistent whitespace.
- Backdoor and Backdoors are preserved as separate
  original category values pending verification.
- Port columns contain decimal and hexadecimal values.
- Invalid, missing, and out-of-range port values are converted to missing values.
- The service value "-" is preserved as a valid categorical value.
## Important Features

- Label: Binary target (0 = normal, 1 = attack).
- attack_cat: Attack category.
- proto: Network protocol.
- state: Connection state.
- dur: Flow duration.
- sbytes / dbytes: Bytes transferred in each direction.

## Preprocessing Considerations

- Normalize categorical whitespace.
- Handle missing values carefully, preserving expected missing attack categories for normal records.
- Convert hexadecimal port values to decimal and validate the port range (0–65535).
- Exclude attack_cat from binary classification features.
- Exclude IP addresses and other identifiers from model inputs.
- Evaluate class imbalance using precision, recall, F1-score,
  and confusion matrix.
