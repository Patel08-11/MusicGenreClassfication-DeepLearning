"""
Quickstart Easy-Way Runner for Music Genre Classification.
Executes the entire end-to-end Deep Learning pipeline in one click:
1. Verifies / Generates dataset audio samples
2. Extracts Mel-Spectrogram features
3. Trains 2D Audio Spectrogram ConvNet with GPU acceleration
4. Evaluates model accuracy & creates Confusion Matrix
5. Runs inference on test songs and produces visual spectrograms
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    SAMPLES_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    DEVICE
)
from dataset.generate_samples import generate_all_samples
from train import train_model
from evaluate import evaluate_model
from predict import predict_genre, print_prediction_results


def run_quickstart(epochs: int = 15, batch_size: int = 16):
    print("\n" + "=" * 70)
    print("      DEEP LEARNING MUSIC GENRE CLASSIFICATION: EASY-WAY PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Project Directory: {PROJECT_ROOT}\n")

    # Step 1: Check or generate audio samples
    wav_files = list(RAW_DATA_DIR.glob("*/*.wav"))
    if len(wav_files) < 30:
        print("[Step 1/5] Synthesizing initial multi-genre audio dataset...")
        generate_all_samples(tracks_per_genre=6, duration=15.0)
    else:
        print(f"[Step 1/5] Found existing dataset with {len(wav_files)} tracks in {RAW_DATA_DIR.name}/")

    # Step 2 & 3: Extract Features & Train Model
    print(f"\n[Step 2 & 3/5] Extracting Mel-Spectrograms and Training 2D ConvNet ({epochs} epochs)...")
    history = train_model(
        model_type="cnn",
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001,
        patience=8,
        force_recompute=False
    )

    # Step 4: Evaluate Model
    print("\n[Step 4/5] Evaluating on Unseen Validation Data...")
    best_model_file = MODELS_DIR / "best_model.pth"
    eval_results = evaluate_model(model_path=best_model_file, model_type="cnn")

    # Step 5: Test Real Inference Predictions
    print("\n[Step 5/5] Running Real-Time Inference on Sample Tracks...")
    demo_files = list(SAMPLES_DIR.glob("*_demo.wav"))
    if not demo_files:
        demo_files = list(RAW_DATA_DIR.glob("*/*.wav"))[:3]

    for demo_audio in demo_files[:3]:
        res = predict_genre(demo_audio, top_k=3, save_spectrogram=True)
        print_prediction_results(res)

    print("\n" + "=" * 70)
    print("[SUCCESS] QUICKSTART RUN COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated Artifacts:")
    print(f"  - Model Checkpoint: {MODELS_DIR / 'best_model.pth'}")
    print(f"  - Training History: {OUTPUTS_DIR / 'training_history.png'}")
    print(f"  - Confusion Matrix: {OUTPUTS_DIR / 'confusion_matrix.png'}")
    print(f"  - Evaluation Report: {OUTPUTS_DIR / 'evaluation_report.txt'}")
    print("\nNext Steps:")
    print("  1. Launch the interactive Web Demo:")
    print("     streamlit run app.py")
    print("  2. Download full 1.2GB GTZAN dataset:")
    print("     python dataset/download_dataset.py")
    print("  3. Predict any custom song:")
    print("     python predict.py --audio path/to/your_song.wav")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_quickstart()
