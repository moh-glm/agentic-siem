"""
Agent 2 implementation — Behaviour & Sequence Analysis Agent.

Uses a trained PyTorch RNN/LSTM model to analyze ordered sequences
of network flows and returns standardized agent responses.
"""

from typing import Any
import numpy as np
import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent


class Agent2(BaseAgent):
    """
    Sequence analysis agent based on Recurrent Neural Networks.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        device: str | None = None,
    ):
        super().__init__(agent_name="agent_2")
        self.model = model
        self.threshold = threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, data: Any) -> dict:
        """
        Analyze an ordered sequence of preprocessed network flows.

        Args:
            data: Either a 2D numpy array/tensor of shape (seq_len, num_features)
                  or a dict containing:
                  - 'sequence': 2D array of shape (seq_len, num_features)
                  - 'input_id': optional string identifier (default: 'sequence_001')

        Returns:
            Standardized agent response dictionary matching docs/interfaces.md.
        """
        input_id = "sequence_001"
        if isinstance(data, dict):
            input_id = str(data.get("input_id", input_id))
            sequence = data.get("sequence")
        else:
            sequence = data

        if sequence is None:
            return self._error_response(input_id, "Missing 'sequence' in input data.")

        try:
            seq_array = np.asarray(sequence, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            return self._error_response(input_id, f"Invalid sequence data: {exc}")

        if seq_array.ndim != 2 or seq_array.shape[0] < 2:
            return self._error_response(
                input_id,
                f"Expected 2D sequence with length >= 2, got shape {seq_array.shape}.",
            )

        sequence_length = int(seq_array.shape[0])
        tensor_x = torch.from_numpy(seq_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_x)
            attack_prob = float(torch.sigmoid(logits).item())

        is_suspicious = attack_prob >= self.threshold
        label = "suspicious_sequence" if is_suspicious else "normal_sequence"
        confidence = attack_prob if is_suspicious else (1.0 - attack_prob)
        confidence = round(min(max(confidence, 0.0), 1.0), 4)

        return self.build_response(
            input_id=input_id,
            prediction={
                "label": label,
                "attack_probability": round(attack_prob, 4),
            },
            confidence=confidence,
            evidence={
                "sequence_length": sequence_length,
                "threshold": self.threshold,
            },
        )

    def _error_response(self, input_id: str, message: str) -> dict:
        """Return a standardized error response per integration rules."""
        return {
            "agent": self.agent_name,
            "status": "error",
            "input_id": input_id,
            "prediction": {},
            "confidence": None,
            "evidence": {"error": message},
        }
