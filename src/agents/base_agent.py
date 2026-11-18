
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Common abstract interface for all SIEM agents.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    @abstractmethod
    def predict(self, data: Any) -> dict:
        """
        Process input data and return a standardized result.
        """
        pass

    def build_response(
        self,
        input_id: str,
        prediction: dict,
        confidence: float | None,
        evidence: dict,
    ) -> dict:
        """
        Build a standardized response shared by all agents.
        """

        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        return {
            "agent": self.agent_name,
            "status": "success",
            "input_id": input_id,
            "prediction": prediction,
            "confidence": confidence,
            "evidence": evidence,
        }
