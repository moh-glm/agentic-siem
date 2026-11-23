
# Agent Communication Interfaces

## 1. Purpose

This document defines the communication contracts between the three agents of the Agentic SIEM project.

All agents return a consistent output structure so that the orchestrator can process their results.

## 2. Common Agent Output

Every agent returns a Python dictionary following this structure:

```python
{
    "agent": "agent_name",
    "status": "success",
    "input_id": "unique_input_identifier",
    "prediction": {},
    "confidence": 0.0,
    "evidence": {}
}
```

### Fields

- `agent`: Name of the agent.
- `status`: Processing status (`success` or `error`).
- `input_id`: Identifier of the analyzed flow or sequence.
- `prediction`: Agent-specific prediction.
- `confidence`: Confidence score between 0.0 and 1.0, when available.
- `evidence`: Additional information supporting the prediction.

When processing fails, agents return `status = "error"`, with `prediction` and `confidence` set to `None` and the error description included in `evidence`.

## 3. Agent 1 — Fast Detection

**Models:** Gaussian Naive Bayes and Logistic Regression.

**Selection strategy:** Both models are evaluated during experimentation. The model with the best validation F1-score is selected and saved. Only this selected model is loaded and executed during runtime.

**Input:** One network flow containing all required Agent 1 features.

The input can be provided as a dictionary or a single-row pandas DataFrame.

**Output example:**

```python
{
    "agent": "agent_1",
    "status": "success",
    "input_id": "flow_001",
    "prediction": {
        "label": "attack",
        "naive_bayes": None,
        "logistic_regression": "attack"
    },
    "confidence": 0.94,
    "evidence": {
        "agreement": None,
        "selected_model": "Logistic Regression"
    }
}
```

### Prediction fields

- `label`: Final classification (`normal` or `attack`).
- `naive_bayes`: Prediction from Gaussian Naive Bayes when it is the selected model; otherwise `None`.
- `logistic_regression`: Prediction from Logistic Regression when it is the selected model; otherwise `None`.
- `selected_model`: Name of the model selected during validation.
- `agreement`: `None`, because only one selected model executes during runtime.

**Important:** The confidence value is the selected model's predicted class probability. It is not necessarily a calibrated probability of correctness.

Agent 1 performs fast binary classification of individual network flows.

## 4. Agent 2 — Sequence Analysis

**Model:** Recurrent Neural Network (RNN).

**Input:** An ordered sequence of network flows or events.

**Output example:**

```python
{
    "agent": "agent_2",
    "status": "success",
    "input_id": "sequence_001",
    "prediction": {
        "label": "suspicious_sequence"
    },
    "confidence": 0.89,
    "evidence": {
        "sequence_length": 10
    }
}
```

The agent analyzes temporal patterns that may not be visible when examining individual flows.

## 5. Agent 3 — Response Decision

**Model:** Decision Tree.

**Input:** Structured detection results and relevant features from the previous agents.

**Possible actions:**

- `MONITOR`
- `ALERT`
- `INVESTIGATE`
- `RECOMMEND_BLOCK`

**Output example:**

```python
{
    "agent": "agent_3",
    "status": "success",
    "input_id": "flow_001",
    "prediction": {
        "action": "ALERT"
    },
    "confidence": 0.91,
    "evidence": {
        "reason": "High attack confidence from detection agents"
    }
}
```

**Important:** The UNSW-NB15 dataset provides traffic classification labels, not mitigation-action labels. The team must define and document a separate response-policy target for training this Decision Tree.

All response actions are recommendations in a simulation. The project must not block real network traffic or modify real systems.

## 6. Orchestrator Responsibilities

The orchestrator coordinates the agents:

1. Receive preprocessed network flows.
2. Send individual flows to Agent 1.
3. Collect Agent 1 results.
4. Construct ordered sequences for Agent 2.
5. Collect Agent 2 results.
6. Send relevant detection information to Agent 3.
7. Collect the recommended response.
8. Forward the combined results to the dashboard.

## 7. Integration Rules

- All agents must follow the common output structure.
- Each agent must identify itself using its `agent` field.
- Confidence scores must be between 0.0 and 1.0 when provided.
- Errors must be reported using `status = "error"`.
- Agents must not directly depend on another agent's internal implementation.
- The orchestrator is responsible for communication and coordination.
- Agent 1 executes only its selected best model during runtime; consumers must not assume both model predictions are available.
