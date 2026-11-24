"""
Tests du module d'évaluation de l'Agent 3, sur des données synthétiques.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.agent3.experiment import (
    load_outputs,
    merge_outputs,
    ratio,
    run_evaluation,
    save_report,
    security_metrics,
    select_threshold,
    validate_outputs,
)


def _synthetic(n, seed, with_category=True):
    """Probabilités bruitées mais corrélées à la vérité terrain."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p1 = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n), 0, 1)
    p2 = np.clip(0.25 + 0.55 * y + rng.normal(0, 0.2, n), 0, 1)
    frame = pd.DataFrame({
        "input_id": [f"f{seed}_{i}" for i in range(n)],
        "p_agent1": p1,
        "p_agent2": p2,
        "y_true": y,
    })
    if with_category:
        categories = rng.choice(["Exploits", "DoS", "Generic", "Analysis"], n)
        frame["attack_cat"] = np.where(y == 1, categories, "-")
    return frame


class TestSecurityMetrics:
    def test_hand_example_exact_values(self):
        actions = ["RECOMMEND_BLOCK", "ALERT", "MONITOR", "MONITOR", "INVESTIGATE", "MONITOR"]
        y = [1, 0, 1, 0, 1, 0]
        m = security_metrics(actions, y)
        assert m["attacks"] == 3 and m["normals"] == 3
        assert m["detection_rate"] == pytest.approx(2 / 3)
        assert m["missed_attack_rate"] == pytest.approx(1 / 3)
        assert m["false_alert_rate"] == pytest.approx(1 / 3)
        assert m["false_block_rate"] == pytest.approx(0.0)
        assert m["block_precision"] == pytest.approx(1.0)
        assert m["blocks"] == 1

    def test_no_blocks_gives_none_precision(self):
        m = security_metrics(["MONITOR", "ALERT"], [0, 1])
        assert m["block_precision"] is None

    def test_no_attacks_gives_none_detection(self):
        m = security_metrics(["MONITOR", "ALERT"], [0, 0])
        assert m["detection_rate"] is None

    def test_ratio_handles_zero_denominator(self):
        assert ratio(1, 0) is None
        assert ratio(1, 4) == pytest.approx(0.25)


class TestValidation:
    def test_valid_frame_passes(self):
        assert len(validate_outputs(_synthetic(20, 1))) == 20

    def test_missing_column_rejected(self):
        frame = _synthetic(10, 1).drop(columns=["p_agent2"])
        with pytest.raises(ValueError):
            validate_outputs(frame)

    def test_probability_out_of_range_rejected(self):
        frame = _synthetic(10, 1)
        frame.loc[0, "p_agent1"] = 1.3
        with pytest.raises(ValueError):
            validate_outputs(frame)

    def test_invalid_label_rejected(self):
        frame = _synthetic(10, 1)
        frame.loc[0, "y_true"] = 2
        with pytest.raises(ValueError):
            validate_outputs(frame)

    def test_duplicate_ids_rejected(self):
        frame = _synthetic(10, 1)
        frame.loc[1, "input_id"] = frame.loc[0, "input_id"]
        with pytest.raises(ValueError):
            validate_outputs(frame)

    def test_load_outputs_reads_csv(self, tmp_path):
        path = tmp_path / "out.csv"
        _synthetic(15, 2).to_csv(path, index=False)
        assert len(load_outputs(path)) == 15


class TestMergeAndThreshold:
    def test_merge_reports_unmatched_flows(self):
        a1 = pd.DataFrame({"input_id": ["a", "b", "c"], "p_agent1": [0.1, 0.2, 0.3], "y_true": [0, 1, 0]})
        a2 = pd.DataFrame({"input_id": ["b", "c", "d"], "p_agent2": [0.5, 0.6, 0.7]})
        merged, report = merge_outputs(a1, a2)
        assert list(merged["input_id"]) == ["b", "c"]
        assert report == {"matched": 2, "agent1_only": 1, "agent2_only": 1}

    def test_select_threshold_separates_classes(self):
        scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        labels = [0, 0, 0, 1, 1, 1]
        threshold = select_threshold(scores, labels)
        assert 0.1 < threshold <= 0.9


@pytest.fixture(scope="module")
def report():
    return run_evaluation(_synthetic(3000, 10), _synthetic(3000, 11), max_depth=4)


class TestRunEvaluation:
    def test_report_has_expected_sections(self, report):
        for key in ("thresholds", "tree", "test_metrics", "depth_comparison", "tree_rules", "severity_test_tree_actions"):
            assert key in report
        assert set(report["test_metrics"]) >= {
            "agent_3_arbre", "politique_sans_arbre", "toujours_MONITOR", "agent_1_seul", "agent_2_seul",
        }

    def test_thresholds_selected_and_in_grid(self, report):
        assert 0.0 < report["thresholds"]["t1"] < 1.0
        assert 0.0 < report["thresholds"]["t2"] < 1.0
        assert report["thresholds"]["t1_t2_selected_on"] == "validation_f1"
        assert report["thresholds"]["high_selected_on"] == "validation_false_block_cap"

    def test_fidelity_high_on_test(self, report):
        assert report["tree"]["fidelity_test"] > 0.9

    def test_monitor_baseline_detects_nothing(self, report):
        assert report["test_metrics"]["toujours_MONITOR"]["detection_rate"] == 0.0

    def test_depth_comparison_covers_depths(self, report):
        depths = [row["max_depth"] for row in report["depth_comparison"]]
        assert depths == [2, 3, 4, 5]

    def test_severity_present_only_with_category(self, report):
        assert report["severity_test_tree_actions"] is not None
        without = run_evaluation(_synthetic(500, 20, with_category=False), _synthetic(500, 21, with_category=False))
        assert without["severity_test_tree_actions"] is None

    def test_report_is_json_serializable(self, report):
        json.dumps(report, default=str)

    def test_save_report_writes_files(self, report, tmp_path):
        path = save_report(report, tmp_path)
        assert path.exists()
        assert (tmp_path / "agent3_tree_rules.txt").read_text(encoding="utf-8") == report["tree_rules"]


class TestHighSelection:
    def _frame_normals_at_09(self):
        """100 normaux à 0,9 et 100 attaques à 0,99 : bloquer à high <= 0,9 crée des faux blocages."""
        normals = pd.DataFrame({
            "input_id": [f"n{i}" for i in range(100)],
            "p_agent1": 0.9, "p_agent2": 0.9, "y_true": 0,
        })
        attacks = pd.DataFrame({
            "input_id": [f"a{i}" for i in range(100)],
            "p_agent1": 0.99, "p_agent2": 0.99, "y_true": 1,
        })
        return pd.concat([normals, attacks], ignore_index=True)

    def test_select_high_respects_false_block_cap(self):
        from src.agents.agent3.experiment import select_high
        assert select_high(self._frame_normals_at_09(), 0.5, 0.5) == 0.95

    def test_high_sweep_blocks_decrease_as_high_increases(self):
        from src.agents.agent3.experiment import high_sweep
        sweep = high_sweep(self._frame_normals_at_09(), 0.5, 0.5)
        blocks = [row["blocks"] for row in sweep]
        assert blocks == sorted(blocks, reverse=True)
        assert len(sweep) == 5

    def test_run_evaluation_reports_how_high_was_chosen(self):
        report = run_evaluation(self._frame_normals_at_09(), self._frame_normals_at_09(), max_depth=2)
        assert report["thresholds"]["high_selected_on"] == "validation_false_block_cap"
        assert report["thresholds"]["high"] in (0.8, 0.85, 0.9, 0.95, 0.99)
