"""
Tests de l'export des sorties de l'agent 3, sans torch ni modèles entraînés.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.agent3.experiment import run_evaluation, validate_outputs
from src.agents.agent3.export_outputs import (
    export_split,
    load_labelled_flows,
    order_for_windows,
    score_agent2_windows,
    window_end_positions,
)

FEATURES = ["x1", "x2"]


def _split(n_per_file=12, seed=0, files=("UNSW-NB15_1_processed.csv", "UNSW-NB15_2_processed.csv")):
    rng = np.random.default_rng(seed)
    frames = []
    for name in files:
        y = rng.integers(0, 2, n_per_file)
        frame = pd.DataFrame({
            "source_file": name,
            "row_in_file": np.arange(n_per_file),
            "Stime": np.sort(rng.random(n_per_file)),
            "Label": y,
            "attack_cat": np.where(y == 1, "DoS", "-"),
            "x1": np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n_per_file), 0, 1),
            "x2": np.clip(0.25 + 0.55 * y + rng.normal(0, 0.2, n_per_file), 0, 1),
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestWindows:
    def test_end_positions_respect_file_boundaries(self):
        sources = ["a"] * 5 + ["b"] * 5
        ends = window_end_positions(sources, window=3)
        assert list(ends) == [2, 3, 4, 7, 8, 9]

    def test_no_window_crosses_files(self):
        sources = np.array(["a"] * 7 + ["b"] * 6 + ["c"] * 4)
        ends = window_end_positions(sources, window=4)
        for end in ends:
            assert len(set(sources[end - 3:end + 1])) == 1

    def test_order_sorts_by_file_then_time(self):
        frame = pd.DataFrame({
            "source_file": ["b", "a", "a"],
            "Stime": [1.0, 5.0, 2.0],
            "row_in_file": [0, 1, 2],
        })
        ordered = order_for_windows(frame)
        assert list(ordered["source_file"]) == ["a", "a", "b"]
        assert list(ordered["Stime"]) == [2.0, 5.0, 1.0]

    def test_agent2_scorer_receives_the_right_windows(self):
        features = np.arange(20, dtype=float).reshape(10, 2)
        ends = np.array([3, 9])
        scores = score_agent2_windows(
            features, ends, window=3,
            scorer=lambda windows: windows[:, -1, 0],
        )
        assert list(scores) == [features[3, 0], features[9, 0]]

    def test_agent2_scoring_is_chunked_without_changing_results(self):
        features = np.arange(40, dtype=float).reshape(20, 2)
        ends = window_end_positions(["a"] * 20, window=4)
        scorer = lambda windows: windows[:, -1, 1]
        full = score_agent2_windows(features, ends, 4, scorer, chunk=1000)
        chunked = score_agent2_windows(features, ends, 4, scorer, chunk=3)
        assert np.allclose(full, chunked)


class TestExportSplit:
    @pytest.fixture
    def exported(self):
        split = _split()
        agent1 = lambda features: features["x1"].to_numpy()
        preprocess = lambda features: features[FEATURES].to_numpy()
        agent2 = lambda windows: windows[:, -1, 1]
        table, report = export_split(split, agent1, preprocess, agent2, feature_columns=FEATURES, window=4)
        return split, table, report

    def test_table_is_valid_for_evaluation(self, exported):
        _, table, _ = exported
        validate_outputs(table)

    def test_report_counts_excluded_flows(self, exported):
        split, table, report = exported
        assert report["flows_in_split"] == len(split)
        assert report["exported"] == len(table)
        assert report["excluded_without_window"] == len(split) - len(table)
        # 3 premiers flux de chaque fichier n'ont pas 3 prédécesseurs dans leur fichier
        assert report["excluded_without_window"] == 6

    def test_input_ids_are_unique_and_traceable(self, exported):
        _, table, _ = exported
        assert table["input_id"].is_unique
        assert table["input_id"].str.contains(":").all()

    def test_labels_and_categories_are_carried(self, exported):
        _, table, _ = exported
        assert set(table["y_true"]) <= {0, 1}
        assert "attack_cat" in table.columns

    def test_agent1_score_is_aligned_with_its_flow(self, exported):
        split, table, _ = exported
        ordered = order_for_windows(split)
        position = ordered.set_index(["source_file", "row_in_file"])
        for _, row in table.head(20).iterrows():
            source, row_number = row["input_id"].rsplit(":", 1)
            flow = position.loc[(source, int(row_number))]
            assert row["p_agent1"] == pytest.approx(flow["x1"])


class TestLoading:
    def test_load_adds_source_and_position(self, tmp_path):
        names = ["UNSW-NB15_1_processed.csv", "UNSW-NB15_2_processed.csv"]
        from src.data.preprocess import get_feature_columns
        columns = get_feature_columns()
        rng = np.random.default_rng(3)
        for name in names:
            frame = pd.DataFrame(rng.random((5, len(columns))), columns=columns)
            frame["Label"] = rng.integers(0, 2, 5)
            frame["attack_cat"] = "-"
            frame["Stime"] = np.arange(5, dtype=float)
            for column in ("sport", "dsport"):
                frame[column] = 80.0
            for column in ("proto", "state", "service"):
                frame[column] = "tcp"
            frame.to_csv(tmp_path / name, index=False)

        df = load_labelled_flows(tmp_path, names)
        assert len(df) == 10
        assert set(df["source_file"]) == set(names)
        assert df["Stime"].is_monotonic_increasing
        assert {"source_file", "row_in_file"} <= set(df.columns)


def test_exported_tables_feed_the_evaluation():
    """Chaîne complète : export -> évaluation, sans torch."""
    validation = _split(n_per_file=400, seed=1)
    test = _split(n_per_file=400, seed=2)
    agent1 = lambda features: features["x1"].to_numpy()
    preprocess = lambda features: features[FEATURES].to_numpy()
    agent2 = lambda windows: windows[:, -1, 1]

    val_table, _ = export_split(validation, agent1, preprocess, agent2, feature_columns=FEATURES, window=4)
    test_table, _ = export_split(test, agent1, preprocess, agent2, feature_columns=FEATURES, window=4)

    report = run_evaluation(val_table, test_table, max_depth=3)
    assert report["sizes"]["test"] == len(test_table)
    assert report["tree"]["fidelity_test"] > 0.8
