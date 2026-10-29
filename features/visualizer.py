"""
Audio and Spectrogram Visualizer Module.
Generates publication-quality charts for:
- Waveform & Mel-Spectrogram & MFCC plots
- Multi-feature inspection dashboards
- Confusion matrices and training loss/accuracy curves
"""

import sys
from typing import Optional, List
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all servers/GUIs
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    OUTPUTS_DIR,
    GENRES
)
from features.extractor import (
    compute_mel_spectrogram,
    compute_mfcc,
    compute_chroma,
    compute_stft_spectrogram
)


# Dark sleek theme styling for modern visualization
plt.style.use("dark_background")
ACCENT_COLOR = "#00E5FF"
GRID_COLOR = "#2A2E3D"


def plot_audio_dashboard(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    title: str = "Audio Spectral Analysis",
    save_path: Optional[str | Path] = None
) -> Path:
    """
    Plot a comprehensive 4-panel dashboard:
    1. Raw Waveform
    2. Mel-Spectrogram
    3. MFCC Heatmap
    4. Chromagram (Pitch Classes)
    """
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    fig.patch.set_facecolor("#0F1117")

    for ax in axes:
        ax.set_facecolor("#161923")
        ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # 1. Waveform
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    axes[0].plot(time_axis, y, color=ACCENT_COLOR, alpha=0.85, linewidth=0.8)
    axes[0].set_title(f"{title} - Raw Waveform", fontsize=11, fontweight="bold", color="#FFFFFF", pad=6)
    axes[0].set_ylabel("Amplitude", fontsize=9, color="#A0AEC0")
    axes[0].set_xlim([0, time_axis[-1]])

    # 2. Mel Spectrogram
    mel_db = compute_mel_spectrogram(y, sr=sr, normalize=False)
    img_mel = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma"
    )
    axes[1].set_title("Log Mel-Spectrogram (Frequency Distribution over Time)", fontsize=11, fontweight="bold", color="#FFFFFF", pad=6)
    axes[1].set_ylabel("Hz (Mel Scale)", fontsize=9, color="#A0AEC0")
    cbar1 = fig.colorbar(img_mel, ax=axes[1], format="%+2.0f dB", pad=0.015)
    cbar1.ax.tick_params(labelsize=8, colors="#A0AEC0")

    # 3. MFCCs
    mfccs = compute_mfcc(y, sr=sr)
    img_mfcc = librosa.display.specshow(
        mfccs,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=axes[2],
        cmap="coolwarm"
    )
    axes[2].set_title("Mel-Frequency Cepstral Coefficients (MFCC - Timbre Representation)", fontsize=11, fontweight="bold", color="#FFFFFF", pad=6)
    axes[2].set_ylabel("MFCC Index", fontsize=9, color="#A0AEC0")
    cbar2 = fig.colorbar(img_mfcc, ax=axes[2], pad=0.015)
    cbar2.ax.tick_params(labelsize=8, colors="#A0AEC0")

    # 4. Chromagram
    chroma = compute_chroma(y, sr=sr)
    img_chroma = librosa.display.specshow(
        chroma,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="chroma",
        ax=axes[3],
        cmap="cividis"
    )
    axes[3].set_title("Chromagram (Harmonic & Pitch Content)", fontsize=11, fontweight="bold", color="#FFFFFF", pad=6)
    axes[3].set_xlabel("Time (seconds)", fontsize=9, color="#A0AEC0")
    cbar3 = fig.colorbar(img_chroma, ax=axes[3], pad=0.015)
    cbar3.ax.tick_params(labelsize=8, colors="#A0AEC0")

    if save_path is None:
        save_path = OUTPUTS_DIR / "audio_spectral_dashboard.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def plot_spectrogram_only(
    mel_db: np.ndarray,
    sr: int = SAMPLE_RATE,
    genre_name: str = "",
    save_path: Optional[str | Path] = None
) -> Path:
    """
    Plot and save an isolated high-resolution Mel-spectrogram.
    """
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#161923")

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )
    title = f"Mel-Spectrogram {f'[{genre_name.upper()}]' if genre_name else ''}"
    ax.set_title(title, fontsize=12, fontweight="bold", color="#FFFFFF")
    ax.set_xlabel("Time (s)", color="#A0AEC0")
    ax.set_ylabel("Frequency (Hz)", color="#A0AEC0")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors="#A0AEC0")

    if save_path is None:
        save_path = OUTPUTS_DIR / f"spectrogram_{genre_name or 'sample'}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def plot_training_history(
    history: dict,
    save_path: Optional[str | Path] = None
) -> Path:
    """
    Plot training and validation loss & accuracy side-by-side.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("#0F1117")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161923")
        ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#00E5FF", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#FF5252", linewidth=2, linestyle="--")
    ax1.set_title("Loss over Epochs (Cross-Entropy)", fontsize=11, fontweight="bold", color="#FFFFFF")
    ax1.set_xlabel("Epoch", color="#A0AEC0")
    ax1.set_ylabel("Loss", color="#A0AEC0")
    ax1.legend(facecolor="#1A202C", edgecolor="none")

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#69F0AE", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#FFD740", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy over Epochs (%)", fontsize=11, fontweight="bold", color="#FFFFFF")
    ax2.set_xlabel("Epoch", color="#A0AEC0")
    ax2.set_ylabel("Accuracy (%)", color="#A0AEC0")
    ax2.legend(facecolor="#1A202C", edgecolor="none")

    if save_path is None:
        save_path = OUTPUTS_DIR / "training_history.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = GENRES,
    save_path: Optional[str | Path] = None
) -> Path:
    """
    Plot a normalized confusion matrix heatmap for all genres.
    """
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#161923")

    # Normalize rows to percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors="#A0AEC0")
    cbar.set_label("Accuracy (%)", color="#A0AEC0")

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Music Genre Confusion Matrix (%)",
        ylabel="True Genre",
        xlabel="Predicted Genre"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color="#E2E8F0")
    plt.setp(ax.get_yticklabels(), color="#E2E8F0")
    ax.title.set_color("#FFFFFF")
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_color("#A0AEC0")
    ax.yaxis.label.set_color("#A0AEC0")

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            color = "#000000" if val > thresh else "#FFFFFF"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    if save_path is None:
        save_path = OUTPUTS_DIR / "confusion_matrix.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path
