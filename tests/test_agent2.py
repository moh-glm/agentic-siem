"""
Unit tests for Agent 2 (Sequence Construction, Input Validation, Model Loading, and Predictions).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.agent2.agent import Agent2
from src.agents.agent2.models import build_rnn_model
from src.agents.agent2.sequence_builder import (
    create_sequences,
    create_sequences_per_file,
    sort_by_time,
)


def test_sort_by_time_preserves_chronological_order():
    df = pd.DataFrame({"Stime": [300, 100, 200], "feat": [3.0, 1.0, 2.0]})
    sorted_df = sort_by_time(df, time_col="Stime")
    assert sorted_df["Stime"].tolist() == [100, 200, 300]
    assert sorted_df["feat"].tolist() == [1.0, 2.0, 3.0]


def test_sequence_labeling_rule_any_attack_in_window():
    X = np.zeros((6, 4), dtype=np.float32)
    # Window 0: [0,0,0] -> 0; Window 1: [0,0,1] -> 1; Window 2: [0,1,0] -> 1; Window 3: [1,0,0] -> 1
    y = np.array([0, 0, 0, 1, 0, 0], dtype=np.int64)
    X_seq, y_seq = create_sequences(X, y, sequence_length=3, stride=1)
    assert X_seq.shape == (4, 3, 4)
    assert y_seq.tolist() == [0, 1, 1, 1]


def test_per_file_sequences_never_cross_file_boundaries():
    # File 1 has 5 rows of all 1s; File 2 has 5 rows of all 9s
    X1 = np.ones((5, 3), dtype=np.float32)
    y1 = np.zeros(5, dtype=np.int64)

    X2 = np.full((5, 3), 9.0, dtype=np.float32)
    y2 = np.ones(5, dtype=np.int64)

    X_seq, y_seq = create_sequences_per_file([X1, X2], [y1, y2], sequence_length=4, stride=1)

    # 2 windows from file 1 + 2 windows from file 2 = 4 windows total (no mixed windows)
    assert X_seq.shape == (4, 4, 3)
    assert y_seq.tolist() == [0, 0, 1, 1]
    assert np.all(X_seq[:2] == 1.0)
    assert np.all(X_seq[2:] == 9.0)


def test_sequence_builder_rejects_invalid_labels_and_nans():
    X = np.ones((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        create_sequences(X, np.array([0, 1, 2, 0, 1]), sequence_length=3)

    X_nan = X.copy()
    X_nan[1, 0] = np.nan
    with pytest.raises(ValueError):
        create_sequences(X_nan, np.zeros(5), sequence_length=3)


def test_agent2_validation_and_checkpoint_loading(tmp_path):
    with pytest.raises(ValueError):
        Agent2(model=build_rnn_model("rnn", input_size=4), threshold=1.5)

    model = build_rnn_model("lstm", input_size=4, hidden_size=16, num_layers=1)
    ckpt_path = tmp_path / "agent2_test.pt"
    torch.save(
        {
            "model_type": "lstm",
            "input_size": 4,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "sequence_length": 5,
            "threshold": 0.5,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    agent = Agent2.from_checkpoint(ckpt_path, device="cpu")

    # Valid input
    valid_seq = np.random.randn(5, 4).astype(np.float32)
    res = agent.predict({"input_id": "seq_ok", "sequence": valid_seq})
    assert res["status"] == "ok"
    assert res["agent"] == "agent_2"
    assert res["prediction"]["label"] in {"normal_sequence", "suspicious_sequence"}

    # Invalid NaN input
    nan_seq = valid_seq.copy()
    nan_seq[0, 0] = np.nan
    err_nan = agent.predict({"input_id": "seq_nan", "sequence": nan_seq})
    assert err_nan["status"] == "error"

    # Invalid feature dimension
    wrong_dim = np.random.randn(5, 6).astype(np.float32)
    err_dim = agent.predict({"input_id": "seq_dim", "sequence": wrong_dim})
    assert err_dim["status"] == "error"
