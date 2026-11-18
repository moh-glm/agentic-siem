# Agentic SIEM — System Architecture

**Project:** Chasse Autonome aux Cyber-Menaces (SIEM Agentique)  
**Dataset:** UNSW-NB15 (proposed)  
**Language:** Python  
**Status:** Initial architecture — Version 1.0

---

## 1. Project Overview

The objective of this project is to design and implement an intelligent Security Information and Event Management (SIEM) system using multiple machine-learning agents.

The system analyzes network traffic data, identifies potentially malicious activities, studies suspicious event sequences, and recommends appropriate security responses.

The architecture integrates three machine-learning agents:

1. **Agent 1 — Traffic Classification:** Naive Bayes and Logistic Regression.
2. **Agent 2 — Sequence Analysis:** Recurrent Neural Network (RNN).
3. **Agent 3 — Response Decision:** Decision Tree.

A central orchestrator coordinates the agents and a dashboard displays the results.

## 2. System Architecture

The system follows this processing pipeline:

1. Dataset ingestion.
2. Data preprocessing and feature preparation.
3. Initial traffic classification by Agent 1.
4. Suspicious event sequence analysis by Agent 2.
5. Threat assessment and response recommendation by Agent 3.
6. Result presentation through the SIEM dashboard.

### Main Components

| Component | Responsibility |
|---|---|
| Dataset | Provides network traffic records |
| Preprocessing | Cleans, transforms and prepares data |
| Agent 1 | Classifies network flows |
| Agent 2 | Detects suspicious temporal patterns |
| Agent 3 | Recommends mitigation actions |
| Orchestrator | Coordinates agents and manages information flow |
| Dashboard | Displays alerts, predictions and recommendations |

## 3. Agent Responsibilities

### 3.1 Agent 1 — Fast Traffic Classification

**Algorithms:**
- Naive Bayes
- Logistic Regression

**Technology:** Scikit-learn

**Responsibilities:**
- Receive preprocessed network traffic records.
- Classify individual network flows as normal or potentially malicious.
- Calculate prediction probabilities or confidence scores.
- Produce structured detection results.

**Expected output:**

```python
{
    "flow_id": "flow_001",
    "prediction": "malicious",
    "confidence": 0.94,
    "source_agent": "agent_1"
}
```

The output will also identify which classifier produced the prediction.

### 3.2 Agent 2 — Suspicious Sequence Analysis

**Algorithm:** Recurrent Neural Network (RNN)

**Technology:** PyTorch

**Responsibilities:**
- Receive ordered sequences of network events or flows.
- Analyze relationships between successive events.
- Identify suspicious temporal patterns.
- Estimate the likelihood of malicious behavior across a sequence.
- Produce a sequence-level threat assessment.

**Expected output:**

```python
{
    "sequence_id": "sequence_001",
    "prediction": "suspicious",
    "confidence": 0.91,
    "source_agent": "agent_2"
}
```

The sequence construction strategy and event ordering must be defined during data preparation.

### 3.3 Agent 3 — Intelligent Response Decision

**Algorithm:** Decision Tree

**Technology:** Scikit-learn

**Responsibilities:**
- Receive structured detection results from the other agents.
- Evaluate threat severity and relevant characteristics.
- Select an appropriate recommended mitigation action.
- Explain the decision using interpretable decision rules.

**Possible actions:**
- `MONITOR`: Continue monitoring the traffic.
- `ALERT`: Generate a security alert.
- `INVESTIGATE`: Request further analysis.
- `RECOMMEND_BLOCK`: Recommend blocking a suspicious source.

**Expected output:**

```python
{
    "incident_id": "incident_001",
    "severity": "high",
    "recommended_action": "RECOMMEND_BLOCK",
    "source_agent": "agent_3"
}
```

**Important training consideration:** UNSW-NB15 provides attack labels, not mitigation-action labels. Therefore, the Decision Tree needs a separately defined response-policy target or a documented rule-based method to generate training labels. It must not treat attack labels as response actions.

## 4. Central Orchestrator

The orchestrator coordinates the three agents.

Its responsibilities include:

- Managing the processing pipeline.
- Passing structured data between agents.
- Combining predictions.
- Managing errors and missing results.
- Creating unified security incidents.
- Sending final results to the dashboard.

The orchestrator must not be tightly coupled to the internal implementation of any agent.

Each agent must expose a consistent interface so that its implementation can be tested independently.

## 5. Data Flow

The proposed processing sequence is:

```text
UNSW-NB15 Dataset
       |
       v
Data Preprocessing
       |
       v
Agent 1: Naive Bayes + Logistic Regression
       |
       v
Initial Traffic Classification
       |
       v
Sequence Construction
       |
       v
Agent 2: RNN
       |
       v
Sequence-Level Threat Assessment
       |
       v
Agent 3: Decision Tree
       |
       v
Recommended Mitigation Action
       |
       v
Central Orchestrator
       |
       v
SIEM Dashboard
```

The implementation may process records in batches. The exact interaction between the orchestrator and the agents will be finalized during integration.

## 6. Common Agent Interface

To simplify integration, each agent must receive structured input and return structured output.

The proposed Python interface is:

```python
class BaseAgent:
    def predict(self, data):
        raise NotImplementedError
```

Each agent will implement its own prediction logic while respecting the common interface.

All outputs should include:
- A unique flow, sequence or incident identifier.
- The prediction or recommended action.
- A confidence score where applicable.
- The agent identifier.

## 7. Dataset

The proposed dataset is **UNSW-NB15**.

It contains normal network traffic and multiple categories of network attacks.

It will be used for:
- Training and evaluating the traffic classifiers.
- Constructing sequences for the RNN.
- Preparing the information required for response decisions.

Dataset preprocessing must avoid data leakage between training, validation and test sets.

The sequence construction method and feature selection strategy must be documented.

## 8. Technologies

| Purpose | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Classical machine learning | Scikit-learn |
| Deep learning | PyTorch |
| Dashboard | Streamlit |
| Version control | Git and GitHub |

## 9. Team Organization

The project is developed by three students.

| Member | Assigned responsibility |
|---|---|
| Member 1 | Agent 1: Naive Bayes and Logistic Regression |
| Member 2 | Agent 2: RNN sequence analysis |
| Member 3 | Agent 3: Decision Tree and dashboard/integration |

All members must follow the common interfaces and coordinate changes through the shared GitHub repository.

## 10. Security and Execution Constraints

The system is designed for cybersecurity analysis and educational demonstration.

It may:
- Detect suspicious traffic.
- Generate security alerts.
- Recommend mitigation actions.
- Simulate response decisions.

It must not automatically block real network traffic or modify real systems.

## 11. Evaluation

Each agent will be evaluated independently and the integrated system will be evaluated as a whole.

Relevant classification metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Additional evaluation will consider:
- RNN sequence detection performance.
- Decision Tree response-policy correctness.
- Processing time.
- Successful communication between agents.

The final report will document experimental results and limitations.

---

**Document version:** 1.0  
