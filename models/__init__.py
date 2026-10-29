"""
Deep Learning Model Architectures for Music Genre Classification.
"""

from .cnn_classifier import AudioSpectrogramCNN
from .crnn_classifier import AudioSpectrogramCRNN

__all__ = ["AudioSpectrogramCNN", "AudioSpectrogramCRNN"]
