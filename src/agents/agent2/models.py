"""
Recurrent Neural Network (RNN & LSTM) architectures for Agent 2.

Analyzes ordered sequences of network flows to detect temporal attack patterns.
Includes checkpoint & preprocessing pipeline loading utilities for Coordinator / Agent 3 integration.
"""

from pathlib import Path
from typing import Any
import joblib
import torch
import torch.nn as nn


class SequenceRNN(nn.Module):
    """Vanilla Recurrent Neural Network for sequence binary classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out).squeeze(-1)
        return logits


class SequenceLSTM(nn.Module):
    """Long Short-Term Memory (LSTM) network for sequence binary classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out).squeeze(-1)
        return logits


def build_rnn_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory function to instantiate an RNN or LSTM model."""
    if input_size < 1:
        raise ValueError(f"input_size must be >= 1, got {input_size}.")
    normalized = model_type.strip().lower()
    if normalized == "rnn":
        return SequenceRNN(input_size, hidden_size, num_layers, dropout)
    if normalized == "lstm":
        return SequenceLSTM(input_size, hidden_size, num_layers, dropout)
    raise ValueError(f"Unsupported model_type: {model_type}. Choose 'rnn' or 'lstm'.")


def save_agent2_bundle(
    checkpoint_path: str | Path,
    model: nn.Module,
    model_type: str,
    input_size: int,
    sequence_length: int,
    preprocessor: Any | None = None,
    threshold: float = 0.5,
    best_epoch: int | None = None,
) -> tuple[Path, Path | None]:
    """
    Save the trained PyTorch model checkpoint together with its fitted
    preprocessing pipeline (saved alongside as a .joblib artifact).
    """
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    preproc_path: Path | None = None
    if preprocessor is not None:
        preproc_path = ckpt_path.with_name(f"{ckpt_path.stem}_preprocessor.joblib")
        joblib.dump(preprocessor, preproc_path)

    torch.save(
        {
            "model_type": model_type,
            "input_size": int(input_size),
            "hidden_size": int(getattr(model, "hidden_size", 64)),
            "num_layers": int(getattr(model, "num_layers", 2)),
            "dropout": float(getattr(model, "dropout", 0.2)),
            "sequence_length": int(sequence_length),
            "threshold": float(threshold),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "preprocessor_filename": preproc_path.name if preproc_path else None,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path, preproc_path


def load_trained_sequence_model(
    checkpoint_path: str | Path,
    preprocessor_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict, Any | None]:
    """
    Load a trained SequenceRNN or SequenceLSTM checkpoint and its fitted
    preprocessing pipeline from disk.

    Returns:
        (model, metadata, preprocessor) tuple with model set to eval() mode.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    resolved_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)

    required_keys = {"model_type", "input_size", "state_dict"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")

    model = build_rnn_model(
        model_type=checkpoint["model_type"],
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint.get("hidden_size", 64)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    model.eval()

    preprocessor = None
    resolved_preproc_path: Path | None = None
    if preprocessor_path is not None:
        resolved_preproc_path = Path(preprocessor_path)
    elif checkpoint.get("preprocessor_filename"):
        candidate = path.with_name(checkpoint["preprocessor_filename"])
        if candidate.exists():
            resolved_preproc_path = candidate

    if resolved_preproc_path is not None:
        if not resolved_preproc_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {resolved_preproc_path}")
        preprocessor = joblib.load(resolved_preproc_path)

    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, metadata, preprocessor
