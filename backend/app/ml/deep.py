"""Compact temporal regression architectures; inputs must carry real time steps."""

import torch
from torch import nn


class TemporalRegressor(nn.Module):
    def __init__(self, model_type: str, sequence_channels: int, static_features: int,
                 timesteps: int, hidden: int = 32):
        super().__init__()
        self.model_type = model_type
        if model_type == "LSTM":
            self.encoder = nn.LSTM(sequence_channels, hidden, batch_first=True)
        elif model_type == "CNN-LSTM":
            self.conv = nn.Sequential(nn.Conv1d(sequence_channels, hidden, 3, padding=1),
                                      nn.ReLU())
            self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        elif model_type == "CNN-Transformer":
            self.conv = nn.Sequential(nn.Conv1d(sequence_channels, hidden, 3, padding=1),
                                      nn.ReLU())
            self.position = nn.Parameter(torch.zeros(1, timesteps, hidden))
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=64,
                                               dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1,
                                                 enable_nested_tensor=False)
        else:
            raise ValueError(f"Unsupported temporal model: {model_type}")
        self.head = nn.Sequential(nn.Linear(hidden + static_features, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if self.model_type == "LSTM":
            encoded, _ = self.encoder(sequence)
            pooled = encoded[:, -1, :]
        else:
            encoded = self.conv(sequence.transpose(1, 2)).transpose(1, 2)
            if self.model_type == "CNN-LSTM":
                encoded, _ = self.encoder(encoded)
                pooled = encoded[:, -1, :]
            else:
                pooled = self.encoder(encoded + self.position).mean(dim=1)
        return self.head(torch.cat((pooled, static), dim=1)).squeeze(1)
