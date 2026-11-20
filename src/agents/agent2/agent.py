"""
Agent 2 implementation — Behaviour & Sequence Analysis Agent.

Uses a trained PyTorch RNN/LSTM model to analyze ordered sequences
of network flows and returns standardized agent responses.

Sequence-Labeling Rule:
    A sequence window is classified as suspicious/attack if at least one flow
    in its window is an attack (probability >= threshold).
"""

from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent
from src.agents.agent2.models import load_trained_sequence_model


class Agent2(BaseAgent):
    """
    Sequence analysis agent based on Recurrent Neural Networks.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        expected_input_size: int | None = None,
        expected_sequence_length: int | None = None,
        device: str | None = None,
    ):
        super().__init__(agent_name="agent_2")
        if not (0.0 < float(threshold) < 1.0):
            raise ValueError(
                f"threshold must be strictly between 0.0 and 1.0, got {threshold}."
            )
        if expected_input_size is not None and expected_input_size < 1:
            raise ValueError(
                f"expected_input_size must be >= 1, got {expected_input_size}."
            )
        if expected_sequence_length is not None and expected_sequence_length < 2:
            raise ValueError(
                f"expected_sequence_length must be >= 2, got {expected_sequence_length}."
            )

        self.model = model
        self.threshold = float(threshold)
        self.expected_input_size = (
            expected_input_size
            if expected_input_size is not None
            else getattr(model, "input_size", None)
        )
        self.expected_sequence_length = expected_sequence_length
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        threshold: float | None = None,
        device: str | None = None,
    ) -> "Agent2":
        """
        Model-loading utility for Coordinator / Agent 3 to instantiate Agent2
        directly from a saved checkpoint file.
        """
        model, metadata = load_trained_sequence_model(checkpoint_path, device=device)
        resolved_threshold = (
            threshold if threshold is not None else float(metadata.get("threshold", 0.5))
        )
        return cls(
            model=model,
            threshold=resolved_threshold,
            expected_input_size=int(metadata["input_size"]),
            expected_sequence_length=metadata.get("sequence_length"),
            device=device,
        )

    def predict(self, data: Any) -> dict:
        """
        Analyze an ordered sequence of preprocessed network flows.
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

        if not np.all(np.isfinite(seq_array)):
            return self._error_response(
                input_id,
                "Input sequence contains NaN or Inf values.",
            )

        if (
            self.expected_input_size is not None
            and seq_array.shape[1] != self.expected_input_size
        ):
            return self._error_response(
                input_id,
                f"Feature dimension mismatch: expected {self.expected_input_size}, got {seq_array.shape[1]}.",
            )

        if (
            self.expected_sequence_length is not None
            and seq_array.shape[0] != self.expected_sequence_length
        ):
            return self._error_response(
                input_id,
                f"Sequence length mismatch: expected {self.expected_sequence_length}, got {seq_array.shape[0]}.",
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
                "labeling_rule": "attack_if_any_flow_in_window_is_attack",
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
