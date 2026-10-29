"""
2D Deep Convolutional Neural Network for Audio Mel-Spectrogram Classification.
Treats audio spectrograms as single-channel acoustic images.
Features:
- 4 Convolutional blocks with Batch Normalization, ReLU, and Max Pooling
- Spatial Dropout to prevent overfitting
- Adaptive Average Pooling for variable length audio tolerance
- Dense classifier head with Dropout
"""

import torch
import torch.nn as nn
from typing import Tuple


class AudioSpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(AudioSpectrogramCNN, self).__init__()

        # Conv Block 1: Input (B, 1, 128, 130) -> (B, 32, 64, 65)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5)
        )

        # Conv Block 2: (B, 32, 64, 65) -> (B, 64, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5)
        )

        # Conv Block 3: (B, 64, 32, 32) -> (B, 128, 16, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )

        # Conv Block 4: (B, 128, 16, 16) -> (B, 256, 8, 8)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(p=dropout_rate)
        )

        # Fully Connected Classifier
        # Flattened dimension: 256 channels * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 1.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, 1, Mel_bins, Time_frames)
        Returns:
            logits: Tensor of shape (Batch, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Test forward pass with expected Spectrogram dimensions
    dummy_input = torch.randn(4, 1, 128, 130)
    model = AudioSpectrogramCNN(num_classes=10)
    output = model(dummy_input)
    print("AudioSpectrogramCNN initialized successfully!")
    print(f"Input shape: {dummy_input.shape} -> Output logits shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
