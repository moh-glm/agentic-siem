"""
Agent 3 : recommandation de réponse à partir des sorties des agents 1 et 2.

Sortie conforme à docs/interfaces.md. L'Agent 3 recommande seulement :
il ne bloque jamais de trafic réel.
"""

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.agent3.features import (
    FEATURE_NAMES,
    build_feature_row,
    decision_confidence,
)
from src.agents.agent3.model import load_model, simplified_rules
from src.agents.agent3.policy import (
    ACTIONS,
    DEFAULT_HIGH,
    DEFAULT_T1,
    DEFAULT_T2,
    check_threshold,
    choose_action,
)


class Agent3(BaseAgent):
    """Agent de décision : un arbre entraîné prédit l'action à partir des deux agents."""

    def __init__(self, model, t1=DEFAULT_T1, t2=DEFAULT_T2, high=DEFAULT_HIGH):
        super().__init__(agent_name="agent_3")
        if model is None:
            raise ValueError("Un modèle entraîné est requis")
        self.model = model
        self.t1 = check_threshold("t1", t1)
        self.t2 = check_threshold("t2", t2)
        self.high = check_threshold("high", high)

    @classmethod
    def from_model_file(cls, path, t1=DEFAULT_T1, t2=DEFAULT_T2, high=DEFAULT_HIGH):
        return cls(load_model(path), t1=t1, t2=t2, high=high)

    def predict(self, data) -> dict:
        """
        data doit contenir :
            input_id : identifiant du flux analysé (le même pour les deux agents)
            agent1_result : sortie de l'agent 1
            agent2_result : sortie de l'agent 2
        """
        if not isinstance(data, dict):
            return self._error_response("unknown", "data doit être un dictionnaire")

        input_id = str(data.get("input_id", "unknown"))
        agent1 = data.get("agent1_result")
        agent2 = data.get("agent2_result")

        if not isinstance(agent1, dict) or not isinstance(agent2, dict):
            return self._error_response(input_id, "agent1_result et agent2_result sont requis")

        if agent1.get("status") != "success" or agent2.get("status") != "success":
            return self._error_response(input_id, "An input agent failed")

        if agent1.get("input_id") != input_id or agent2.get("input_id") != input_id:
            return self._error_response(input_id, "Input ID mismatch")

        try:
            row = build_feature_row(agent1, agent2)
        except ValueError as exc:
            return self._error_response(input_id, str(exc))

        features = pd.DataFrame([row], columns=FEATURE_NAMES)

        try:
            action = str(self.model.predict(features)[0])
        except ValueError as exc:
            return self._error_response(input_id, f"Prédiction impossible : {exc}")

        if action not in ACTIONS:
            return self._error_response(input_id, f"Action inconnue : {action}")

        return self.build_response(
            input_id=input_id,
            prediction={"action": action},
            confidence=round(decision_confidence(row["p_agent1"], row["p_agent2"]), 4),
            evidence={
                "features": row,
                "rules": simplified_rules(self.model, features),
                "policy_action": choose_action(
                    row["p_agent1"], row["p_agent2"], self.t1, self.t2, self.high
                ),
                "thresholds": {"t1": self.t1, "t2": self.t2, "high": self.high},
            },
        )

    def _error_response(self, input_id: str, message: str) -> dict:
        return {
            "agent": self.agent_name,
            "status": "error",
            "input_id": input_id,
            "prediction": {},
            "confidence": None,
            "evidence": {"error": message},
        }
