"""
Audio Genre Inference and Prediction Script.
Runs the trained deep learning model on any audio track (WAV, MP3, etc.).
Features:
- Segment-level inference with soft-voting aggregation across the entire song
- Top-K genre predictions with confidence percentages
- Audio summary diagnostics (BPM, spectral centroid, energy)
- Generates and saves visual spectrogram of the predicted track
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEVICE,
    GENRES,
    NUM_CLASSES,
    MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLE_RATE
)
from features.extractor import (
    load_audio,
    extract_track_features,
    extract_audio_summary,
    compute_mel_spectrogram
)
from features.visualizer import plot_audio_dashboard, plot_spectrogram_only
from models.cnn_classifier import AudioSpectrogramCNN
from models.crnn_classifier import AudioSpectrogramCRNN


def load_trained_model(
    model_path: Path | str = MODELS_DIR / "best_model.pth",
    model_type: str = "cnn"
) -> torch.nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please train a model first using 'python train.py'!")

    if model_type.lower() == "crnn":
        model = AudioSpectrogramCRNN(num_classes=NUM_CLASSES)
    else:
        model = AudioSpectrogramCNN(num_classes=NUM_CLASSES)

    checkpoint = torch.load(str(model_path), map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    return model


def predict_genre(
    audio_path: str | Path,
    model: torch.nn.Module | None = None,
    model_type: str = "cnn",
    top_k: int = 3,
    save_spectrogram: bool = True
) -> Dict[str, Any]:
    """
    Classify an audio file into musical genres.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model is None:
        model = load_trained_model(model_type=model_type)

    # 1. Extract feature tensors for all 3s segments
    batch_tensors, waveform = extract_track_features(audio_path, sr=SAMPLE_RATE, max_duration=30.0)
    batch_tensors = batch_tensors.to(DEVICE)

    # 2. Forward pass across all segments
    with torch.no_grad():
        logits = model(batch_tensors)
        probs_per_segment = F.softmax(logits, dim=1).cpu().numpy()

    # 3. Soft-voting aggregation: average probability vectors over all time slices
    avg_probs = np.mean(probs_per_segment, axis=0)

    # 4. Rank genres
    ranked_indices = np.argsort(avg_probs)[::-1]
    top_predictions: List[Dict[str, Any]] = []
    for idx in ranked_indices[:top_k]:
        top_predictions.append({
            "genre": GENRES[idx],
            "confidence": float(avg_probs[idx]),
            "percentage": round(float(avg_probs[idx] * 100), 2)
        })

    primary_genre = top_predictions[0]["genre"]

    # 5. Extract acoustic summary diagnostics
    summary = extract_audio_summary(waveform, sr=SAMPLE_RATE)

    # 6. Generate visual spectrogram
    dashboard_path = None
    if save_spectrogram:
        out_name = f"predict_{audio_path.stem}.png"
        dashboard_path = plot_audio_dashboard(
            waveform,
            sr=SAMPLE_RATE,
            title=f"Predicted Genre: {primary_genre.upper()} ({top_predictions[0]['percentage']}%)",
            save_path=OUTPUTS_DIR / out_name
        )

    return {
        "file": str(audio_path),
        "primary_genre": primary_genre,
        "confidence": top_predictions[0]["percentage"],
        "top_predictions": top_predictions,
        "all_probabilities": {GENRES[i]: round(float(avg_probs[i] * 100), 2) for i in range(NUM_CLASSES)},
        "audio_summary": summary,
        "spectrogram_image": str(dashboard_path) if dashboard_path else None
    }


def print_prediction_results(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"TRACK: {Path(result['file']).name}")
    print(f"PREDICTED GENRE: >> {result['primary_genre'].upper()} << ({result['confidence']}%)")
    print("=" * 60)

    print("\nConfidence Breakdown:")
    for i, item in enumerate(result["top_predictions"], 1):
        bar_len = int(item["percentage"] / 4)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {i}. {item['genre']:<12} [{bar}] {item['percentage']:5.1f}%")

    print("\nAcoustic Diagnostics:")
    for k, v in result["audio_summary"].items():
        print(f"  - {k.replace('_', ' ').title():<24}: {v}")

    if result.get("spectrogram_image"):
        print(f"\nSaved Spectrogram Analysis: {result['spectrogram_image']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Music Genre from Audio Track")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (wav, mp3, etc.)")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "crnn"], help="Model type")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to display")
    args = parser.parse_args()

    res = predict_genre(args.audio, model_type=args.model, top_k=args.top_k)
    print_prediction_results(res)
