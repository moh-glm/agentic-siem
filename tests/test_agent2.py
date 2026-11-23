"""
Unit tests for Agent 2 (Sequence Construction, Raw-Flow Inference,
Preprocessor Persistence, Status Alignment, Edge Cases, and Predictions).
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.agent1.preprocessing import build_preprocessor
from src.agents.agent2.agent import Agent2
from src.agents.agent2.models import build_rnn_model, save_agent2_bundle
from src.agents.agent2.sequence_builder import (
    create_sequences,
    create_sequences_per_file,
    sort_by_time,
)
from src.data.dataset_loader import COLUMNS
from src.data.preprocess import preprocess_dataset


def _make_sample_raw_df(num_rows: int = 10) -> pd.DataFrame:
    """Create a synthetic raw UNSW-NB15-compatible DataFrame for end-to-end tests."""
    rng = np.random.default_rng(42)
    data = {}
    for col in COLUMNS:
        if col in {"srcip", "dstip"}:
            data[col] = ["192.168.1.10"] * num_rows
        elif col in {"proto", "state", "service"}:
            data[col] = ["tcp"] * num_rows
        elif col == "attack_cat":
            data[col] = ["Normal"] * num_rows
        elif col == "Label":
            data[col] = [0] * num_rows
        elif col == "Stime":
            data[col] = list(range(1000 + num_rows, 1000, -1))
        else:
            data[col] = rng.uniform(0.0, 10.0, size=num_rows)
    return pd.DataFrame(data)


def test_sort_by_time_preserves_chronological_order():
    df = pd.DataFrame({"Stime": [300, 100, 200], "feat": [3.0, 1.0, 2.0]})
    sorted_df = sort_by_time(df, time_col="Stime")
    assert sorted_df["Stime"].tolist() == [100, 200, 300]
    assert sorted_df["feat"].tolist() == [1.0, 2.0, 3.0]


def test_sequence_labeling_rule_any_attack_in_window():
    X = np.zeros((6, 4), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 0, 0], dtype=np.int64)
    X_seq, y_seq = create_sequences(X, y, sequence_length=3, stride=1)
    assert X_seq.shape == (4, 3, 4)
    assert y_seq.tolist() == [0, 1, 1, 1]


def test_per_file_sequences_never_cross_file_boundaries():
    X1 = np.ones((5, 3), dtype=np.float32)
    y1 = np.zeros(5, dtype=np.int64)
    X2 = np.full((5, 3), 9.0, dtype=np.float32)
    y2 = np.ones(5, dtype=np.int64)

    X_seq, y_seq = create_sequences_per_file([X1, X2], [y1, y2], sequence_length=4, stride=1)
    assert X_seq.shape == (4, 4, 3)
    assert y_seq.tolist() == [0, 0, 1, 1]
    assert np.all(X_seq[:2] == 1.0)
    assert np.all(X_seq[2:] == 9.0)


def test_sequence_builder_edge_cases():
    X = np.ones((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        create_sequences(X, np.array([0, 1, 2, 0, 1]), sequence_length=3)

    X_nan = X.copy()
    X_nan[1, 0] = np.nan
    with pytest.raises(ValueError):
        create_sequences(X_nan, np.zeros(5), sequence_length=3)

    with pytest.raises(ValueError):
        create_sequences(X, np.zeros(5), sequence_length=10)

    with pytest.raises(ValueError):
        create_sequences(X, sequence_length=1)
    with pytest.raises(ValueError):
        create_sequences(X, sequence_length=3, stride=0)


def test_agent2_bundle_saving_and_raw_flow_to_sequence_inference(tmp_path):
    raw_train = _make_sample_raw_df(num_rows=12)
    clean_train = preprocess_dataset(raw_train)
    preprocessor = build_preprocessor(scale_numerical=True, dense_output=True)
    X_train_2d = preprocessor.fit_transform(clean_train)
    input_size = X_train_2d.shape[1]

    model = build_rnn_model("lstm", input_size=input_size, hidden_size=16, num_layers=1)
    ckpt_path, preproc_path = save_agent2_bundle(
        checkpoint_path=tmp_path / "agent2_bundle.pt",
        model=model,
        model_type="lstm",
        input_size=input_size,
        sequence_length=10,
        preprocessor=preprocessor,
        threshold=0.5,
        best_epoch=1,
    )
    assert ckpt_path.exists()
    assert preproc_path is not None and preproc_path.exists()

    agent = Agent2.from_checkpoint(ckpt_path, device="cpu")
    assert agent.preprocessor is not None

    raw_test = _make_sample_raw_df(num_rows=10)
    res = agent.predict({"input_id": "raw_seq_01", "raw_flows": raw_test})
    assert res["status"] == "success"
    assert res["agent"] == "agent_2"
    assert res["input_id"] == "raw_seq_01"
    assert res["prediction"]["label"] in {"normal_sequence", "suspicious_sequence"}
    assert 0.0 <= res["confidence"] <= 1.0


def test_agent2_edge_cases_and_status_consistency():
    with pytest.raises(ValueError):
        Agent2(model=build_rnn_model("rnn", input_size=4), threshold=0.0)
    with pytest.raises(ValueError):
        Agent2(model=build_rnn_model("rnn", input_size=4), threshold=1.0)

    agent = Agent2(
        model=build_rnn_model("rnn", input_size=4, hidden_size=8, num_layers=1),
        expected_input_size=4,
        expected_sequence_length=5,
        device="cpu",
    )

    valid_seq = np.random.randn(5, 4).astype(np.float32)
    res_ok = agent.predict({"input_id": "seq_ok", "sequence": valid_seq})
    assert res_ok["status"] == "success"

    err_short = agent.predict(np.random.randn(1, 4).astype(np.float32))
    assert err_short["status"] == "error"

    err_len = agent.predict(np.random.randn(3, 4).astype(np.float32))
    assert err_len["status"] == "error"

    nan_seq = valid_seq.copy()
    nan_seq[0, 0] = np.inf
    err_inf = agent.predict(nan_seq)
    assert err_inf["status"] == "error"

    err_dim = agent.predict(np.random.randn(5, 7).astype(np.float32))
    assert err_dim["status"] == "error"

    err_no_preproc = agent.predict_from_raw_flows(_make_sample_raw_df(5))
    assert err_no_preproc["status"] == "error"
