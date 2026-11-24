"""
Tests unitaires de l'Agent 3 : politique, features, arbre, sauvegarde et agent.

Les sorties de l'agent 1 reprennent le format réel de src/agents/agent1/agent.py :
un seul modèle sélectionné est exécuté, et confidence est la probabilité de la classe prédite.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.agents.agent3.agent import Agent3
from src.agents.agent3.features import (
    FEATURE_NAMES,
    attack_probability_from_agent1,
    build_feature_row,
    decision_confidence,
)
from src.agents.agent3.model import decision_rules, load_model, save_model, train_decision_tree
from src.agents.agent3.policy import ACTIONS, apply_policy, choose_action


def _agent1(input_id, p, status="success", selected="Logistic Regression"):
    """Sortie de l'agent 1 : label et confiance du modèle sélectionné."""
    if status != "success":
        return {
            "agent": "agent_1",
            "status": "error",
            "input_id": input_id,
            "prediction": None,
            "confidence": None,
            "evidence": {"error": "Test error"},
        }
    label = "attack" if p >= 0.5 else "normal"
    confidence = p if label == "attack" else 1.0 - p
    return {
        "agent": "agent_1",
        "status": "success",
        "input_id": input_id,
        "prediction": {
            "label": label,
            "naive_bayes": label if selected == "Gaussian Naive Bayes" else None,
            "logistic_regression": label if selected == "Logistic Regression" else None,
        },
        "confidence": confidence,
        "evidence": {"agreement": None, "selected_model": selected},
    }


def _agent2(input_id, p, status="success"):
    return {
        "agent": "agent_2",
        "status": status,
        "input_id": input_id,
        "prediction": {
            "label": "suspicious_sequence" if p >= 0.5 else "normal_sequence",
            "attack_probability": p,
        },
        "confidence": max(p, 1 - p),
    }


@pytest.fixture(scope="module")
def policy_grid():
    """Grille régulière de probabilités, étiquetée par la politique."""
    grid = np.round(np.linspace(0.0, 1.0, 41), 3)
    features = pd.DataFrame([{"p_agent1": a, "p_agent2": b} for a in grid for b in grid])
    targets = apply_policy(features)
    return features, targets


@pytest.fixture(scope="module")
def trained_tree(policy_grid):
    features, targets = policy_grid
    return train_decision_tree(features, targets, max_depth=6)


class TestPolicy:
    @pytest.mark.parametrize(
        "p_agent1, p_agent2, expected",
        [
            (0.10, 0.10, "MONITOR"),
            (0.20, 0.95, "INVESTIGATE"),
            (0.80, 0.30, "INVESTIGATE"),
            (0.70, 0.70, "ALERT"),
            (0.95, 0.97, "RECOMMEND_BLOCK"),
        ],
    )
    def test_each_action_case(self, p_agent1, p_agent2, expected):
        assert choose_action(p_agent1, p_agent2) == expected

    def test_threshold_boundary_counts_as_detection(self):
        assert choose_action(0.5, 0.1) == "INVESTIGATE"
        assert choose_action(0.9, 0.9) == "RECOMMEND_BLOCK"

    def test_block_requires_both_agents_very_sure(self):
        assert choose_action(0.99, 0.85) == "ALERT"

    @pytest.mark.parametrize("bad", [-0.1, 1.2, float("nan"), "abc", None])
    def test_invalid_probability_rejected(self, bad):
        with pytest.raises(ValueError):
            choose_action(bad, 0.5)

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            choose_action(0.5, 0.5, t1=0.0)

    def test_apply_policy_on_empty_frame(self):
        result = apply_policy(pd.DataFrame(columns=FEATURE_NAMES))
        assert len(result) == 0


class TestFeatures:
    def test_agent1_attack_label_uses_confidence_directly(self):
        assert attack_probability_from_agent1(_agent1("f", 0.8)) == pytest.approx(0.8)

    def test_agent1_normal_label_uses_complement(self):
        result = {
            "agent": "agent_1",
            "status": "success",
            "input_id": "f",
            "prediction": {"label": "normal", "naive_bayes": "normal", "logistic_regression": None},
            "confidence": 0.9,
            "evidence": {"agreement": None, "selected_model": "Gaussian Naive Bayes"},
        }
        assert attack_probability_from_agent1(result) == pytest.approx(0.1)

    def test_agent1_selected_naive_bayes_is_accepted(self):
        result = _agent1("f", 0.9, selected="Gaussian Naive Bayes")
        assert attack_probability_from_agent1(result) == pytest.approx(0.9)

    def test_agent1_unknown_label_rejected(self):
        bad = _agent1("f", 0.8)
        bad["prediction"]["label"] = "suspicious"
        with pytest.raises(ValueError):
            attack_probability_from_agent1(bad)

    def test_agent1_missing_confidence_rejected(self):
        bad = _agent1("f", 0.8)
        bad["confidence"] = None
        with pytest.raises(ValueError):
            attack_probability_from_agent1(bad)

    def test_agent2_reads_attack_probability(self):
        assert build_feature_row(_agent1("f", 0.8), _agent2("f", 0.3)) == {
            "p_agent1": pytest.approx(0.8),
            "p_agent2": pytest.approx(0.3),
        }

    def test_agent2_missing_probability_rejected(self):
        bad = _agent2("f", 0.3)
        del bad["prediction"]["attack_probability"]
        with pytest.raises(ValueError):
            build_feature_row(_agent1("f", 0.8), bad)

    def test_wrong_agent_name_rejected(self):
        with pytest.raises(ValueError):
            build_feature_row(_agent2("f", 0.8), _agent2("f", 0.3))

    @pytest.mark.parametrize(
        "p_agent1, p_agent2, expected",
        [(0.9, 0.95, 0.9), (0.2, 0.95, 0.8), (0.5, 0.5, 0.5)],
    )
    def test_decision_confidence_is_weakest_certainty(self, p_agent1, p_agent2, expected):
        assert decision_confidence(p_agent1, p_agent2) == pytest.approx(expected)


class TestTreeAndRules:
    def test_tree_fits_policy_grid_exactly(self, trained_tree, policy_grid):
        features, targets = policy_grid
        assert (trained_tree.predict(features) == targets.to_numpy()).all()

    def test_tree_reproduces_policy_on_clear_points(self, trained_tree):
        for p1, p2 in [(0.05, 0.05), (0.2, 0.95), (0.7, 0.7), (0.95, 0.97)]:
            features = pd.DataFrame([{"p_agent1": p1, "p_agent2": p2}])
            assert trained_tree.predict(features)[0] == choose_action(p1, p2)

    def test_decision_rules_follow_the_path(self, trained_tree):
        features = pd.DataFrame([{"p_agent1": 0.95, "p_agent2": 0.97}])
        rules = decision_rules(trained_tree, features)
        assert rules
        for rule in rules:
            assert rule.startswith(("p_agent1", "p_agent2"))
            assert " <= " in rule or " > " in rule

    def test_save_and_load_roundtrip(self, trained_tree, tmp_path):
        path = tmp_path / "agent3.joblib"
        save_model(trained_tree, path)
        loaded = load_model(path)
        features = pd.DataFrame([{"p_agent1": 0.2, "p_agent2": 0.95}])
        assert loaded.predict(features)[0] == trained_tree.predict(features)[0]

    def test_load_rejects_wrong_file(self, tmp_path):
        path = tmp_path / "bad.joblib"
        joblib.dump({"not": "a tree"}, path)
        with pytest.raises(ValueError):
            load_model(path)


class TestAgent3:
    @pytest.fixture
    def agent(self, trained_tree):
        return Agent3(trained_tree)

    def test_success_output_structure(self, agent):
        result = agent.predict({
            "input_id": "flow_001",
            "agent1_result": _agent1("flow_001", 0.95),
            "agent2_result": _agent2("flow_001", 0.97),
        })
        for key in ("agent", "status", "input_id", "prediction", "confidence", "evidence"):
            assert key in result
        assert result["agent"] == "agent_3"
        assert result["status"] == "success"
        assert result["input_id"] == "flow_001"
        assert result["prediction"]["action"] == "RECOMMEND_BLOCK"
        assert 0.5 <= result["confidence"] <= 1.0
        assert result["evidence"]["policy_action"] == "RECOMMEND_BLOCK"
        assert isinstance(result["evidence"]["rules"], list)

    def test_action_always_allowed(self, agent):
        points = [(0.0, 0.0), (0.2, 0.9), (0.6, 0.6), (0.99, 0.99), (1.0, 0.0)]
        for i, (p1, p2) in enumerate(points):
            result = agent.predict({
                "input_id": f"f{i}",
                "agent1_result": _agent1(f"f{i}", p1),
                "agent2_result": _agent2(f"f{i}", p2),
            })
            assert result["status"] == "success"
            assert result["prediction"]["action"] in ACTIONS

    def test_failed_input_agent_returns_error(self, agent):
        result = agent.predict({
            "input_id": "flow_002",
            "agent1_result": _agent1("flow_002", 0.9, status="error"),
            "agent2_result": _agent2("flow_002", 0.9),
        })
        assert result["status"] == "error"
        assert result["confidence"] is None
        assert result["prediction"] == {}

    def test_input_id_mismatch_returns_error(self, agent):
        result = agent.predict({
            "input_id": "flow_001",
            "agent1_result": _agent1("flow_001", 0.9),
            "agent2_result": _agent2("flow_002", 0.9),
        })
        assert result["status"] == "error"
        assert result["evidence"]["error"] == "Input ID mismatch"

    def test_missing_probability_returns_error_not_crash(self, agent):
        bad = _agent2("f", 0.9)
        del bad["prediction"]["attack_probability"]
        result = agent.predict({
            "input_id": "f",
            "agent1_result": _agent1("f", 0.9),
            "agent2_result": bad,
        })
        assert result["status"] == "error"

    def test_out_of_range_confidence_returns_error(self, agent):
        bad = _agent1("f", 0.9)
        bad["confidence"] = 1.4
        result = agent.predict({
            "input_id": "f",
            "agent1_result": bad,
            "agent2_result": _agent2("f", 0.9),
        })
        assert result["status"] == "error"

    def test_non_dict_input_returns_error(self, agent):
        result = agent.predict(None)
        assert result["status"] == "error"

    def test_missing_model_rejected(self):
        with pytest.raises(ValueError):
            Agent3(None)

    def test_invalid_threshold_rejected_at_init(self, trained_tree):
        with pytest.raises(ValueError):
            Agent3(trained_tree, t1=1.5)


class TestSimplifiedRules:
    def test_each_variable_appears_at_most_once(self, trained_tree):
        from src.agents.agent3.model import simplified_rules
        features = pd.DataFrame([{"p_agent1": 0.7, "p_agent2": 0.95}])
        rules = simplified_rules(trained_tree, features)
        assert rules
        for name in ("p_agent1", "p_agent2"):
            assert sum(rule.startswith(name) or f" {name} " in rule for rule in rules) <= 1
        assert len(rules) <= 2

    def test_bounded_rule_format(self, trained_tree):
        from src.agents.agent3.model import simplified_rules
        features = pd.DataFrame([{"p_agent1": 0.7, "p_agent2": 0.95}])
        for rule in simplified_rules(trained_tree, features):
            assert " <= " in rule or " > " in rule
