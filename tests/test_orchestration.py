"""
Tests de l'orchestrateur minimal, avec des agents factices au format commun.
"""

import numpy as np
import pytest

from src.orchestration.coordinator import Coordinator


class FakeAgent1:
    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def predict(self, data):
        self.seen.append(data["input_id"])
        if self.fail:
            return {"agent": "agent_1", "status": "error", "input_id": data["input_id"],
                    "prediction": None, "confidence": None, "evidence": {"error": "boom"}}
        p = data["p"]
        label = "attack" if p >= 0.5 else "normal"
        return {"agent": "agent_1", "status": "success", "input_id": data["input_id"],
                "prediction": {"label": label}, "confidence": p if label == "attack" else 1 - p,
                "evidence": {}}


class FakeAgent2:
    def __init__(self):
        self.seen = []

    def predict(self, data):
        self.seen.append((data["input_id"], np.asarray(data["sequence"]).shape))
        p = float(np.mean(data["sequence"]))
        return {"agent": "agent_2", "status": "success", "input_id": data["input_id"],
                "prediction": {"label": "suspicious_sequence" if p >= 0.5 else "normal_sequence",
                               "attack_probability": p},
                "confidence": max(p, 1 - p), "evidence": {}}


class RecordingAgent3:
    """Agent 3 factice : enregistre ses entrées et renvoie l'action selon les probabilités."""

    def __init__(self):
        self.received = []

    def predict(self, data):
        self.received.append(data)
        ok = data["agent1_result"]["status"] == "success" and data["agent2_result"]["status"] == "success"
        if not ok:
            return {"agent": "agent_3", "status": "error", "input_id": data["input_id"],
                    "prediction": {}, "confidence": None, "evidence": {"error": "An input agent failed"}}
        r1 = data["agent1_result"]
        p1 = r1["confidence"] if r1["prediction"]["label"] == "attack" else 1 - r1["confidence"]
        return {"agent": "agent_3", "status": "success", "input_id": data["input_id"],
                "prediction": {"action": "ALERT" if p1 >= 0.7 else "MONITOR"},
                "confidence": 0.9, "evidence": {}}


def _seq(p):
    return np.full((10, 4), p)


def test_same_input_id_reaches_all_agents():
    a1, a2, a3 = FakeAgent1(), FakeAgent2(), RecordingAgent3()
    coordinator = Coordinator(a1, a2, a3)
    result = coordinator.process_flow("flow_001", {"p": 0.9}, _seq(0.8))
    assert a1.seen == ["flow_001"]
    assert a2.seen == [("flow_001", (10, 4))]
    assert a3.received[0]["input_id"] == "flow_001"
    assert result["decision"]["status"] == "success"


def test_agent3_receives_both_agent_outputs():
    a3 = RecordingAgent3()
    Coordinator(FakeAgent1(), FakeAgent2(), a3).process_flow("f", {"p": 0.2}, _seq(0.1))
    received = a3.received[0]
    assert received["agent1_result"]["agent"] == "agent_1"
    assert received["agent2_result"]["agent"] == "agent_2"


def test_agent1_error_propagates_without_crash():
    result = Coordinator(FakeAgent1(fail=True), FakeAgent2(), RecordingAgent3()).process_flow(
        "f", {"p": 0.9}, _seq(0.9))
    assert result["agent1"]["status"] == "error"
    assert result["decision"]["status"] == "error"


def test_elapsed_time_is_recorded():
    result = Coordinator(FakeAgent1(), FakeAgent2(), RecordingAgent3()).process_flow("f", {"p": 0.9}, _seq(0.9))
    assert result["elapsed_ms"] >= 0


def test_batch_summary_counts_actions_and_errors():
    items = [
        ("f1", {"p": 0.9}, _seq(0.9)),
        ("f2", {"p": 0.1}, _seq(0.1)),
        ("f3", {"p": 0.95}, _seq(0.95)),
    ]
    batch = Coordinator(FakeAgent1(), FakeAgent2(), RecordingAgent3()).process_batch(items)
    summary = batch["summary"]
    assert summary["flows"] == 3
    assert summary["actions"] == {"ALERT": 2, "MONITOR": 1}
    assert summary["errors"] == 0
    assert summary["mean_ms_per_flow"] is not None


def test_batch_counts_errors():
    class FailingForBad(FakeAgent1):
        def predict(self, data):
            if data["input_id"] == "bad":
                return {"agent": "agent_1", "status": "error", "input_id": "bad",
                        "prediction": None, "confidence": None, "evidence": {"error": "boom"}}
            return super().predict(data)

    items = [("ok", {"p": 0.9}, _seq(0.9)), ("bad", {"p": 0.9}, _seq(0.9))]
    batch = Coordinator(FailingForBad(), FakeAgent2(), RecordingAgent3()).process_batch(items)
    assert batch["summary"]["errors"] == 1
    assert batch["summary"]["actions"] == {"ALERT": 1}
