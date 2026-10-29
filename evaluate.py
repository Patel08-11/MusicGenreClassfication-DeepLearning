"""
Evaluation and Metrics Pipeline for Music Genre Classification.
Computes:
- Overall Validation Accuracy & Top-3 Accuracy
- Per-genre Precision, Recall, and F1-score
- Confusion Matrix Heatmap
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEVICE,
    NUM_CLASSES,
    GENRES,
    MODELS_DIR,
    OUTPUTS_DIR
)
from dataset.audio_loader import get_data_loaders
from models.cnn_classifier import AudioSpectrogramCNN
from models.crnn_classifier import AudioSpectrogramCRNN
from features.visualizer import plot_confusion_matrix


def evaluate_model(
    model_path: Path | str = MODELS_DIR / "best_model.pth",
    model_type: str = "cnn"
) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Train the model first using 'python train.py'!")

    print("=" * 65)
    print("Music Genre Classifier Evaluation")
    print(f"Checkpoint: {model_path.resolve()}")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # 1. Load Model
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

    # 2. Load Validation Set
    _, val_loader, (_, n_val) = get_data_loaders(batch_size=32)
    print(f"Evaluating on {n_val} validation audio slices...\n")

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_top3: list[bool] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            top1_preds = torch.argmax(probs, dim=1).cpu().numpy()
            targets_np = targets.numpy()

            all_preds.extend(top1_preds)
            all_targets.extend(targets_np)

            # Top-3 Accuracy
            top3_indices = torch.topk(probs, k=min(3, NUM_CLASSES), dim=1).indices.cpu().numpy()
            for t, top3 in zip(targets_np, top3_indices):
                all_top3.append(bool(t in top3))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    top1_acc = np.mean(all_preds == all_targets) * 100.0
    top3_acc = np.mean(all_top3) * 100.0

    print("=" * 65)
    print(f"RESULTS SUMMARY:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print("=" * 65)

    # Classification Report
    unique_present_classes = np.unique(np.concatenate([all_targets, all_preds]))
    target_names = [GENRES[i] for i in unique_present_classes]
    report = classification_report(all_targets, all_preds, labels=unique_present_classes, target_names=target_names)
    print("\nDetailed Per-Genre Classification Report:\n")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))
    cm_path = plot_confusion_matrix(cm, class_names=GENRES, save_path=OUTPUTS_DIR / "confusion_matrix.png")
    print(f"\nSaved Confusion Matrix Plot to: {cm_path}")

    # Save report to text file
    report_file = OUTPUTS_DIR / "evaluation_report.txt"
    with open(report_file, "w") as f:
        f.write("MUSIC GENRE CLASSIFICATION REPORT\n")
        f.write("=" * 45 + "\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.2f}%\n\n")
        f.write(report)
    print(f"Saved Text Report to: {report_file}")

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "report": report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Music Genre Classification Model")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "crnn"], help="Model type")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pth"), help="Path to checkpoint")
    args = parser.parse_args()

    evaluate_model(model_path=args.checkpoint, model_type=args.model)
