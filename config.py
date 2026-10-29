"""
Global configuration for Music Genre Classification.
Contains audio parameters, dataset definitions, model architecture settings,
and training hyperparameters.
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "saved_models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Standard GTZAN 10 Music Genres
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}
IDX_TO_GENRE = {idx: genre for idx, genre in enumerate(GENRES)}
NUM_CLASSES = len(GENRES)

# Audio Feature Extraction Hyperparameters
SAMPLE_RATE = 22050        # Standard sampling rate for music analysis (Hz)
TRACK_DURATION = 30        # Total track duration in GTZAN (seconds)
SEGMENT_DURATION = 3.0     # Analysis segment duration (seconds) per training slice
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_DURATION)

# Short-Time Fourier Transform (STFT) & Mel-Spectrogram Parameters
N_FFT = 2048               # Window length for FFT
HOP_LENGTH = 512           # Number of audio samples between adjacent STFT columns
N_MELS = 128               # Number of Mel frequency bands
F_MIN = 20.0               # Minimum frequency (Hz)
F_MAX = SAMPLE_RATE / 2.0  # Maximum frequency (Nyquist frequency ~11025 Hz)
N_MFCC = 20                # Number of MFCC coefficients

# Expected Spectrogram Dimensions for 3-second audio slices:
# Time bins: ceil(3.0 * 22050 / 512) = 130
TIME_BINS = int(SAMPLES_PER_SEGMENT / HOP_LENGTH) + 1  # ~130 bins
SPECTROGRAM_SHAPE = (1, N_MELS, TIME_BINS)             # (Channels=1, Freq=128, Time=130)

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
