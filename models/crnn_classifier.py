"""
Convolutional Recurrent Neural Network (CRNN) for Music Genre Classification.
Combines 2D CNN (for timbral/frequency pattern extraction) with
Bidirectional GRU (for rhythmic and temporal progression).
"""

import torch
import torch.nn as nn


class AudioSpectrogramCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        rnn_hidden_size: int = 128,
        num_rnn_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        super(AudioSpectrogramCRNN, self).__init__()

        # Conv front-end: extracts timbre and frequency textures
        self.conv_layers = nn.Sequential(
            # Block 1: (B, 1, 128, T) -> (B, 32, 64, T)
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout2d(dropout_rate * 0.5),

            # Block 2: (B, 32, 64, T) -> (B, 64, 32, T)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout2d(dropout_rate * 0.5),

            # Block 3: (B, 64, 32, T) -> (B, 128, 8, T)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
            nn.Dropout2d(dropout_rate)
        )

        # Adaptive frequency pooling collapses frequency dimension to 1: (B, 128, 1, T)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Bidirectional GRU to model rhythm across time frames
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_rnn_layers > 1 else 0.0
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram tensor (Batch, 1, Mel_bins=128, Time_frames=130)
        Returns:
            logits: (Batch, num_classes)
        """
        # CNN Feature extraction: (B, C, F, T)
        x = self.conv_layers(x)
        x = self.freq_pool(x)  # (B, 128, 1, T)

        # Reshape for RNN: (B, T, Features=128)
        x = x.squeeze(2)       # (B, 128, T)
        x = x.permute(0, 2, 1) # (B, T, 128)

        # RNN Forward pass
        rnn_out, _ = self.rnn(x)  # (B, T, 2 * hidden_size)

        # Temporal average pooling across time frames
        context = torch.mean(rnn_out, dim=1)  # (B, 2 * hidden_size)

        logits = self.classifier(context)
        return logits
