"""
Unit tests for Agent 2 (Sequence Builder, RNN/LSTM models, and Agent2 interface).
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.agent2.agent import Agent2
from src.agents.agent2.models import SequenceLSTM, SequenceRNN, build_rnn_model
from src.agents.agent2.sequence_builder import create_sequences, sort_by_time


def test_sort_by_time():
    df = pd.DataFrame({"Stime": [300, 100, 200], "feature": [3, 1, 2]})
    sorted_df = sort_by_time(df)
    assert sorted_df["Stime"].tolist() == [100, 200, 300]
    assert sorted_df["feature"].tolist() == [1, 2, 3]


def test_create_sequences_shapes_and_labels():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    X_seq, y_seq = create_sequences(X, y, sequence_length=4, stride=2)
    # Windows: [0:4], [2:6], [4:8], [6:10] -> 4 sequences
    assert X_seq.shape == (4, 4, 2)
    assert y_seq.tolist() == [1, 1, 0, 0]


def test_create_sequences_invalid_length():
    X = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        create_sequences(X, sequence_length=1)
    with pytest.raises(ValueError):
        create_sequences(X, sequence_length=10)


def test_build_rnn_and_lstm_models():
    rnn = build_rnn_model("rnn", input_size=8, hidden_size=16, num_layers=1)
    lstm = build_rnn_model("lstm", input_size=8, hidden_size=16, num_layers=2)
    assert isinstance(rnn, SequenceRNN)
    assert isinstance(lstm, SequenceLSTM)

    with pytest.raises(ValueError):
        build_rnn_model("unknown_arch", input_size=8)


def test_agent2_predict_contract():
    model = build_rnn_model("lstm", input_size=5, hidden_size=16, num_layers=1)
    agent = Agent2(model=model, threshold=0.5, device="cpu")

    sequence = np.random.randn(10, 5).astype(np.float32)
    response = agent.predict({"input_id": "seq_test_01", "sequence": sequence})

    assert response["agent"] == "agent_2"
    assert response["status"] == "success"
    assert response["input_id"] == "seq_test_01"
    assert response["prediction"]["label"] in {"suspicious_sequence", "normal_sequence"}
    assert 0.0 <= response["confidence"] <= 1.0
    assert response["evidence"]["sequence_length"] == 10


def test_agent2_error_handling():
    model = build_rnn_model("rnn", input_size=4, hidden_size=8, num_layers=1)
    agent = Agent2(model=model, device="cpu")

    # Sequence too short (length 1 < 2)
    short_seq = np.zeros((1, 4), dtype=np.float32)
    res = agent.predict({"input_id": "seq_err", "sequence": short_seq})
    assert res["status"] == "error"
    assert res["confidence"] is None
