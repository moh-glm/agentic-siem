"""
Recurrent Neural Network (RNN & LSTM) architectures for Agent 2.

Analyzes ordered sequences of network flows to detect temporal attack patterns.
"""

import torch
import torch.nn as nn


class SequenceRNN(nn.Module):
    """
    Vanilla Recurrent Neural Network for sequence binary classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.rnn(x)
        # Take the output from the last time step
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out).squeeze(-1)
        return logits


class SequenceLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for sequence binary classification.
    Captures longer temporal dependencies in network traffic.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output from the last time step
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
    """
    Factory function to instantiate an RNN or LSTM model.
    """
    normalized = model_type.strip().lower()
    if normalized == "rnn":
        return SequenceRNN(input_size, hidden_size, num_layers, dropout)
    if normalized == "lstm":
        return SequenceLSTM(input_size, hidden_size, num_layers, dropout)
    raise ValueError(f"Unsupported model_type: {model_type}. Choose 'rnn' or 'lstm'.")
