"""
Audio Dataset Loader and PyTorch Dataset / DataLoader implementation.
Supports:
1. Scanning directory structure: data/raw/<genre>/*.wav
2. Segmenting tracks into uniform 3-second slices
3. Feature caching into .npz archive for 20x faster training
4. Stratified train/validation splitting
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    GENRES,
    GENRE_TO_IDX,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    BATCH_SIZE,
    RANDOM_STATE
)
from features.extractor import load_audio, extract_segments, audio_to_mel_tensor


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset yielding (spectrogram_tensor, label_idx).
    Spectrogram tensor shape: (1, 128, TIME_BINS)
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        # data shape: (N, 1, 128, TIME_BINS)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def build_or_load_dataset(
    data_dir: Path = RAW_DATA_DIR,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes all audio tracks in data_dir/<genre>/*.wav, slices them into 3s segments,
    computes Mel-spectrograms, and returns (X, y).
    Caches results into a .npz file for instant subsequent loading.
    """
    if cache_path is None:
        cache_path = PROCESSED_DATA_DIR / "genre_spectrograms.npz"
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached spectrogram dataset from: {cache_path}")
        archive = np.load(str(cache_path))
        return archive["X"], archive["y"]

    print(f"Extracting Mel-spectrograms from raw audio in: {data_dir}...")
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for genre in GENRES:
        genre_folder = Path(data_dir) / genre
        if not genre_folder.exists():
            continue

        audio_files = list(genre_folder.glob("*.wav")) + list(genre_folder.glob("*.au")) + list(genre_folder.glob("*.mp3"))
        print(f"  [{genre.upper()}] Found {len(audio_files)} tracks")

        for file_idx, filepath in enumerate(audio_files):
            try:
                y, sr = load_audio(filepath, sr=SAMPLE_RATE, duration=30.0)
                segments = extract_segments(y, sr=sr, segment_duration=SEGMENT_DURATION)
                for seg in segments:
                    tensor = audio_to_mel_tensor(seg, sr=sr)
                    X_list.append(tensor.numpy())
                    y_list.append(GENRE_TO_IDX[genre])
            except Exception as e:
                print(f"    Warning: Could not process {filepath.name}: {e}")

    if len(X_list) == 0:
        raise ValueError(f"No audio files found in {data_dir}. Run 'python dataset/generate_samples.py' or download GTZAN.")

    X = np.stack(X_list, axis=0)  # Shape: (N, 1, 128, TIME_BINS)
    y = np.array(y_list, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, y=y)
    print(f"Processed {len(X)} audio slices across {len(np.unique(y))} classes. Saved cache to: {cache_path}")
    return X, y


def get_data_loaders(
    batch_size: int = BATCH_SIZE,
    test_size: float = 0.2,
    force_recompute: bool = False
) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
    """
    Returns (train_loader, val_loader, (train_count, val_count)).
    Uses stratified split to preserve balanced genre proportions.
    """
    X, y = build_or_load_dataset(force_recompute=force_recompute)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_dataset = SpectrogramDataset(X_train, y_train)
    val_dataset = SpectrogramDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, (len(train_dataset), len(val_dataset))
