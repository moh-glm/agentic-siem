# Agentic SIEM — Autonomous Cyber-Threat Hunting

An academic Data Mining project that explores autonomous cyber-threat detection and response through a multi-agent machine learning architecture.

## Project Overview

The system analyzes network traffic, identifies suspicious behavior, and supports threat-response decisions through specialized agents.

The project uses the **UNSW-NB15** network intrusion dataset.

## Architecture

| Component | Responsibility | Technique |
|---|---|---|
| Agent 1 — Detection | Fast network-flow classification | Gaussian Naive Bayes, Logistic Regression |
| Agent 2 — Behavior Analysis | Analyze sequences of network events | RNN, LSTM |
| Agent 3 — Response | Determine appropriate response actions | Decision Tree |
| Central Coordinator | Coordinate agent communication | Multi-agent integration |
| Monitoring Dashboard | Present alerts and system activity | Streamlit |

## Dataset

- **Primary dataset:** UNSW-NB15
- **Total records:** 2,540,047
- **Features:** 49 columns
- **Normal traffic:** 87.35%
- **Attack traffic:** 12.65%

See [`docs/dataset_analysis.md`](docs/dataset_analysis.md) for dataset analysis, preprocessing, duplicate auditing, split methodology, and Agent 1 experimental results.

## Technologies

- Python
- pandas and NumPy
- scikit-learn
- PyTorch
- Streamlit
- Git and GitHub

## Repository Structure

```text
agentic-siem/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── dataset_analysis.md
│   ├── installation.md
│   └── interfaces.md
├── notebooks/
├── reports/
│   └── results/
│       ├── agent1/
│       └── agent2/
├── src/
│   ├── agents/
│   │   ├── agent1/
│   │   ├── agent2/
│   │   └── agent3/
│   └── data/
├── tests/
├── requirements.txt
└── README.md
```

## Agent 1 — Current Results

Agent 1 has implemented two network-flow classifiers and a group-aware evaluation methodology.

**Final Logistic Regression test results:**

| Metric | Result |
|---|---:|
| Accuracy | 98.76% |
| Precision | 91.29% |
| Recall | 99.88% |
| F1-score | 95.39% |

- Sample size: 500,000 records.
- Split: approximately 70% training, 15% validation, 15% testing.
- Identical model-input feature patterns are kept in the same split.
- Full automated test suite: 29 tests passing.

These results apply to the sampled UNSW-NB15 evaluation and should not be interpreted as guaranteed performance on unseen real-world networks.

## Installation

Clone the repository:

```bash
git clone https://github.com/moh-glm/agentic-siem.git
cd agentic-siem
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the UNSW-NB15 dataset files in the appropriate `data/raw/` directory. Dataset files are not included in Git.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Documentation

- [System Architecture](docs/architecture.md)
- [Dataset Analysis and Agent 1 Results](docs/dataset_analysis.md)
- [Agent Interfaces](docs/interfaces.md)
- [Installation Guide](docs/installation.md)

## Team

Three students — ING4 Computer Science.

## Project Status

- [x] Initial project structure and documentation
- [x] UNSW-NB15 loading and preprocessing
- [x] Agent 1 classification pipelines
- [x] Group-aware data splitting and duplicate audit
- [x] Agent 1 experimental evaluation
- [x] Agent 1 automated tests
- [ ] Final Agent 2 validation and integration
- [ ] Agent 3 implementation and validation
- [ ] Central coordinator integration
- [ ] Monitoring dashboard integration
- [ ] Final system demonstration and technical report