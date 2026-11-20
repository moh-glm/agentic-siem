# UNSW-NB15 Dataset Analysis

## 1. Dataset Overview

- Dataset: UNSW-NB15
- Total records: 2,540,047
- Total columns: 49
- CSV files: 4
- Normal records: 2,218,764
- Attack records: 321,283

## 2. Class Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Normal (0) | 2,218,764 | 87.35% |
| Attack (1) | 321,283 | 12.65% |

The dataset is imbalanced, with normal traffic representing approximately 87.35% of all records. Therefore, accuracy alone is insufficient for evaluating detection performance.

## 3. Data Quality and Preprocessing

- All four CSV files were successfully read.
- All files contain 49 columns.
- Binary labels were validated.
- Attack categories contain inconsistent whitespace.
- Backdoor and Backdoors are preserved as separate original category values pending verification.
- Port columns contain decimal and hexadecimal values.
- Invalid, missing, and out-of-range port values are converted to missing values.
- The service value "-" is preserved as a valid categorical value.
- Categorical values are stripped of surrounding whitespace and normalized.

### Feature Selection

The binary classification target is `Label`:

- `0`: Normal traffic
- `1`: Attack traffic

The model uses 43 candidate input features.

The following columns are excluded from model inputs:

- `Label`: target variable.
- `attack_cat`: attack-category information that would cause target leakage.
- `srcip` and `dstip`: IP address identifiers.
- `Stime` and `Ltime`: timestamps excluded from Agent 1's tabular classification features.

The selected input features contain 40 numerical columns and 3 categorical columns:

- Numerical: processed traffic and connection measurements.
- Categorical: `proto`, `state`, and `service`.

Numerical features are converted to numeric values, missing values are imputed using training-set medians, and categorical missing values are imputed with `"unknown"` before one-hot encoding.

## 4. Duplicate Feature Audit

A full-dataset audit was performed to identify repeated records and repeated model-input patterns.

| Measurement | Records |
|---|---:|
| Total records | 2,540,047 |
| Unique complete rows | 2,059,417 |
| Repeated complete-row occurrences | 480,630 |
| Unique model-input patterns | 2,032,474 |
| Repeated model-input occurrences | 507,573 |
| Conflicting-label feature groups | 0 |

Repeated model-input patterns are important because identical feature patterns appearing in different dataset splits can make evaluation less independent.

No feature group with conflicting binary labels was found during this audit.

## 5. Agent 1 Sampling and Split Methodology

Agent 1 uses a reproducible class-stratified sample of 500,000 records from the processed dataset.

### Sample Distribution

| Class | Sample records |
|---|---:|
| Normal (0) | 436,756 |
| Attack (1) | 63,244 |
| Total | 500,000 |

### Group-Aware Split

A deterministic hash-based splitting method is used instead of ordinary random row splitting.

The feature columns of each record are hashed. Records with identical model-input feature patterns are assigned to the same split.

The resulting split is:

| Split | Records | Percentage |
|---|---:|---:|
| Training | 349,771 | 69.95% |
| Validation | 75,094 | 15.02% |
| Testing | 75,135 | 15.03% |
| Total | 500,000 | 100% |

The split was verified to have zero shared feature hashes between:

- Training and validation.
- Training and testing.
- Validation and testing.

Both classes are present in all three splits, and their distributions remain close to the overall sample distribution.

The split is deterministic for the same feature data and split parameters.

## 6. Agent 1 Machine Learning Models

Two classification models are implemented:

### Gaussian Naive Bayes

- Numerical features are imputed without standard scaling.
- Categorical features are imputed and one-hot encoded.
- Gaussian Naive Bayes is used as a fast baseline classifier.

### Logistic Regression

- Numerical features are imputed and standardized.
- Categorical features are imputed and one-hot encoded.
- `class_weight="balanced"` is used to account for class imbalance.
- Maximum iterations: 1,000.
- Random state: 42.

Both models are trained using the training split and evaluated on the validation split.

The best model is selected using validation F1-score. The test split is reserved for final evaluation.

## 7. Agent 1 Experimental Results

The following results were obtained using the 500,000-record sample and group-aware split.

### Validation Results

| Metric | Gaussian Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 82.22% | 98.85% |
| Precision | 40.90% | 91.86% |
| Recall | 87.23% | 99.85% |
| F1-score | 55.69% | 95.69% |

Logistic Regression achieved the highest validation F1-score and was selected for final testing.

### Final Test Results — Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 98.76% |
| Precision | 91.29% |
| Recall | 99.88% |
| F1-score | 95.39% |

Confusion matrix:

| Actual / Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 64,605 | 916 |
| Attack | 12 | 9,602 |

The model detected 9,602 of the 9,614 attack records in the test split, while missing 12 attacks.

## 8. Evaluation Limitations

The group-aware split prevents identical model-input feature patterns from appearing across different splits. This addresses one source of potential evaluation leakage.

However, several limitations remain:

- The experiment uses a 500,000-record sample rather than the entire dataset.
- Identical feature patterns are not necessarily identical real-world network events.
- Hash-based grouping does not guarantee separation by host, time period, network, or attack campaign.
- The evaluation does not establish performance on future traffic or unseen networks.
- The results depend on the UNSW-NB15 dataset and its sampling characteristics.
- Precision, recall, F1-score, and confusion matrix should be considered alongside accuracy.

Further evaluation using temporal or network-based separation would provide stronger evidence of generalization.

## 9. Important Features

- `Label`: binary target.
- `attack_cat`: attack category, excluded from binary model inputs.
- `proto`: network protocol.
- `state`: connection state.
- `dur`: flow duration.
- `sbytes` / `dbytes`: bytes transferred in each direction.

## 10. Current Status

- [x] UNSW-NB15 dataset loaded and inspected.
- [x] Data cleaning and preprocessing implemented.
- [x] Feature selection implemented.
- [x] Full-dataset duplicate audit completed.
- [x] Reproducible class-stratified sampling implemented.
- [x] Group-aware train/validation/test splitting implemented.
- [x] Gaussian Naive Bayes implemented.
- [x] Logistic Regression implemented.
- [x] Validation and final test evaluation completed.
- [x] Automated tests passing for the current implementation.
