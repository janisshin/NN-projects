import torch
from torch import nn


class SequenceCNN(nn.Module):
    """Two-layer 1D CNN for 150-bp DNA sequence classification."""

    def __init__(
        self,
        input_channels: int = 4,
        sequence_length: int = 150,
        num_filters: int = 30,
        kernel_size: int = 18,
        second_layer_filters: int = 60,
        hidden_size: int = 50,
        output_classes: int = 2,
        conv_dropout: float = 0.1,
        hidden_dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, second_layer_filters, kernel_size=kernel_size),
            nn.BatchNorm1d(second_layer_filters),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            nn.MaxPool1d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, sequence_length)
            flattened_size = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, output_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
