"""
Agent 2 implementation — Behaviour & Sequence Analysis Agent.

Uses a trained PyTorch RNN/LSTM model (and optional fitted preprocessing pipeline)
to analyze ordered sequences of network flows and return standardized agent responses.

Sequence-Labeling Rule:
    A sequence window is classified as suspicious/attack if at least one flow
    in its window is an attack (probability >= threshold).

Standalone CLI usage:
    python -m src.agents.agent2.agent --checkpoint models/agent2_best_model.pt --demo
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent
from src.agents.agent2.models import load_trained_sequence_model
from src.agents.agent2.sequence_builder import sort_by_time
from src.data.preprocess import preprocess_dataset


class Agent2(BaseAgent):
    """Sequence analysis agent based on Recurrent Neural Networks."""

    def __init__(
        self,
        model: nn.Module,
        preprocessor: Any | None = None,
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
        self.preprocessor = preprocessor
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
        preprocessor_path: str | Path | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ) -> "Agent2":
        """
        Model-loading utility for Coordinator / Agent 3 to instantiate Agent2
        directly from a saved checkpoint and its fitted preprocessing pipeline.
        """
        model, metadata, preprocessor = load_trained_sequence_model(
            checkpoint_path=checkpoint_path,
            preprocessor_path=preprocessor_path,
            device=device,
        )
        resolved_threshold = (
            threshold if threshold is not None else float(metadata.get("threshold", 0.5))
        )
        return cls(
            model=model,
            preprocessor=preprocessor,
            threshold=resolved_threshold,
            expected_input_size=int(metadata["input_size"]),
            expected_sequence_length=metadata.get("sequence_length"),
            device=device,
        )

    def predict_from_raw_flows(
        self,
        raw_flows: pd.DataFrame | list[dict[str, Any]],
        input_id: str = "raw_sequence_001",
        sort_chronologically: bool = True,
    ) -> dict:
        """
        End-to-end raw-flow-to-sequence inference:
            1. Clean raw network flow records via `preprocess_dataset`.
            2. Optionally sort chronologically by `Stime`.
            3. Transform features using the fitted `self.preprocessor`.
            4. Run temporal sequence prediction via `self.predict`.
        """
        if self.preprocessor is None:
            return self._error_response(
                input_id,
                "No fitted preprocessor attached to Agent2. Load with preprocessor artifact.",
            )

        try:
            df_raw = (
                raw_flows.copy()
                if isinstance(raw_flows, pd.DataFrame)
                else pd.DataFrame(raw_flows)
            )
        except Exception as exc:
            return self._error_response(input_id, f"Invalid raw flows container: {exc}")

        if df_raw.empty:
            return self._error_response(input_id, "Raw flows input is empty.")

        try:
            df_clean = preprocess_dataset(df_raw)
            if sort_chronologically:
                df_clean = sort_by_time(df_clean, time_col="Stime")
            features_2d = self.preprocessor.transform(df_clean)
            if hasattr(features_2d, "toarray"):
                features_2d = features_2d.toarray()
        except Exception as exc:
            return self._error_response(
                input_id, f"Raw flow preprocessing failed: {exc}"
            )

        return self.predict({"input_id": input_id, "sequence": features_2d})

    def predict(self, data: Any) -> dict:
        """
        Analyze an ordered sequence of network flows.

        Accepts:
            - A 2D numpy array/tensor of preprocessed features (seq_len, num_features)
            - A dict with 'sequence' (2D array) and optional 'input_id'
            - A dict with 'raw_flows' (DataFrame or list of flow dicts) and optional 'input_id'
        """
        input_id = "sequence_001"
        if isinstance(data, dict):
            input_id = str(data.get("input_id", input_id))
            if "raw_flows" in data and data["raw_flows"] is not None:
                return self.predict_from_raw_flows(
                    raw_flows=data["raw_flows"],
                    input_id=input_id,
                    sort_chronologically=bool(data.get("sort_chronologically", True)),
                )
            sequence = data.get("sequence")
        elif isinstance(data, pd.DataFrame):
            return self.predict_from_raw_flows(raw_flows=data, input_id=input_id)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Agent 2 (Sequence Analysis) independently."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/agent2_best_model.pt",
        help="Path to saved Agent 2 model checkpoint (.pt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional decision threshold override in (0.0, 1.0).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a synthetic sanity check sequence through the loaded Agent 2 model.",
    )
    args = parser.parse_args()

    agent = Agent2.from_checkpoint(args.checkpoint, threshold=args.threshold)
    if args.demo:
        seq_len = agent.expected_sequence_length or 10
        feat_dim = agent.expected_input_size or 204
        sample_seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
        result = agent.predict({"input_id": "cli_demo_001", "sequence": sample_seq})
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Agent 2 loaded successfully (input_size={agent.expected_input_size}, "
            f"sequence_length={agent.expected_sequence_length}, "
            f"preprocessor_loaded={agent.preprocessor is not None})."
        )


if __name__ == "__main__":
    main()
