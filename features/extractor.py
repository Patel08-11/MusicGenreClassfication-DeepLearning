"""
Audio Feature Extraction Module.
Converts raw audio waveforms into visual/spectral representations:
- Mel-Scale Spectrogram (Primary input for 2D Deep CNN)
- Linear STFT Spectrogram
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Chromagram (Harmonic / pitch content)
- Acoustic Summary Statistics (Centroid, Rolloff, ZCR, BPM)
"""

import sys
from typing import Tuple, Dict, Any, List
from pathlib import Path
import numpy as np
import librosa
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    N_MFCC,
    SEGMENT_DURATION,
    SAMPLES_PER_SEGMENT,
    TIME_BINS
)


def load_audio(filepath: str | Path, sr: int = SAMPLE_RATE, duration: float | None = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as a mono float32 waveform resampled to the target sample rate.

    Args:
        filepath: Path to audio file (wav, mp3, ogg, flac, etc.)
        sr: Target sampling rate (default 22050 Hz)
        duration: Maximum duration in seconds to load (default all)

    Returns:
        waveform: 1D numpy array with audio samples in [-1.0, 1.0]
        sr: Sampling rate
    """
    filepath = str(filepath)
    y, actual_sr = librosa.load(filepath, sr=sr, mono=True, duration=duration)
    return y.astype(np.float32), actual_sr


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute log-power Mel-spectrogram in decibels.

    Args:
        y: 1D audio waveform
        sr: Sampling rate
        n_fft: FFT window length
        hop_length: Hop length between analysis frames
        n_mels: Number of Mel frequency bands
        normalize: If True, normalize values into [0.0, 1.0]

    Returns:
        mel_db: 2D numpy array of shape (n_mels, time_frames)
    """
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="constant")

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    if normalize:
        # Standardize dB range from [-80 dB, 0 dB] into [0.0, 1.0]
        min_db = -80.0
        mel_db = np.clip(mel_db, min_db, 0.0)
        mel_db = (mel_db - min_db) / (-min_db)

    return mel_db.astype(np.float32)


def compute_stft_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform (STFT) magnitude spectrogram in dB.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return stft_db.astype(np.float32)


def compute_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.astype(np.float32)


def compute_chroma(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Compute 12-dimensional chromagram (pitch class profile).
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    return chroma.astype(np.float32)


def fix_segment_length(y: np.ndarray, target_length: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """
    Ensure audio slice matches exact target sample length by padding or trimming.
    """
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="wrap")
    elif len(y) > target_length:
        y = y[:target_length]
    return y


def extract_segments(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION,
    overlap: float = 0.0
) -> List[np.ndarray]:
    """
    Split a long audio track into uniform segments (e.g., 3-second slices).

    Args:
        y: Full waveform
        sr: Sampling rate
        segment_duration: Slice duration in seconds
        overlap: Overlap fraction between consecutive segments [0.0, 0.9]

    Returns:
        List of 1D numpy arrays, each of length = int(segment_duration * sr)
    """
    segment_len = int(segment_duration * sr)
    hop_len = int(segment_len * (1.0 - overlap))
    total_samples = len(y)

    segments = []
    if total_samples <= segment_len:
        segments.append(fix_segment_length(y, segment_len))
        return segments

    for start in range(0, total_samples - segment_len + 1, hop_len):
        seg = y[start : start + segment_len]
        segments.append(seg)

    # If remainder at end has meaningful audio (> 1.0 sec), pad and include
    remainder = total_samples % hop_len
    if remainder > int(1.0 * sr) and len(segments) > 0:
        tail = y[-segment_len:] if total_samples >= segment_len else y
        segments.append(fix_segment_length(tail, segment_len))

    return segments


def audio_to_mel_tensor(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_time_bins: int = TIME_BINS
) -> torch.Tensor:
    """
    Convert a 1D audio slice into a PyTorch FloatTensor of shape (1, N_MELS, TIME_BINS).
    Pads or truncates in the time dimension to guarantee uniform input dimension for the CNN.
    """
    mel = compute_mel_spectrogram(y, sr=sr, normalize=True)

    # Ensure time dimension matches target_time_bins exactly
    current_time_bins = mel.shape[1]
    if current_time_bins < target_time_bins:
        pad_width = target_time_bins - current_time_bins
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    elif current_time_bins > target_time_bins:
        mel = mel[:, :target_time_bins]

    # Add channel dimension: (1, 128, TIME_BINS)
    tensor = torch.from_numpy(mel).unsqueeze(0).float()
    return tensor


def extract_track_features(
    filepath: str | Path,
    sr: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION,
    max_duration: float = 30.0
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extract a batch of Mel-spectrogram tensors for all 3s slices of an audio track.

    Returns:
        batch_tensor: shape (num_segments, 1, 128, TIME_BINS)
        full_waveform: original 1D waveform
    """
    y, _ = load_audio(filepath, sr=sr, duration=max_duration)
    segments = extract_segments(y, sr=sr, segment_duration=segment_duration)
    tensors = [audio_to_mel_tensor(seg, sr=sr) for seg in segments]
    batch = torch.stack(tensors, dim=0)  # (N, 1, 128, TIME_BINS)
    return batch, y


def extract_audio_summary(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Extract comprehensive summary statistics for waveform visualization & analysis.
    """
    duration = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(y**2)))

    # Spectral centroid (brightness of sound)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))

    # Zero Crossing Rate (percussiveness / noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # Estimated Tempo / BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        bpm = 0.0

    return {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "rms_energy": round(rms, 4),
        "spectral_centroid_hz": round(centroid_mean, 1),
        "spectral_rolloff_hz": round(rolloff_mean, 1),
        "zero_crossing_rate": round(zcr_mean, 4),
        "estimated_bpm": round(bpm, 1)
    }
