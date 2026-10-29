"""
Deep Learning Training Pipeline for Music Genre Classification.
Trains AudioSpectrogramCNN or AudioSpectrogramCRNN on audio Mel-Spectrograms.
Features:
- GPU acceleration with CUDA
- Learning rate scheduler & early stopping
- Best model weight saving & history logging
- Automatic training curve plots
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEVICE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODELS_DIR,
    OUTPUTS_DIR
)
from dataset.audio_loader import get_data_loaders
from models.cnn_classifier import AudioSpectrogramCNN
from models.crnn_classifier import AudioSpectrogramCRNN
from features.visualizer import plot_training_history


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    return val_loss, val_acc


def train_model(
    model_type: str = "cnn",
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = 10,
    force_recompute: bool = False
) -> Dict[str, List[float]]:
    """
    Complete training loop for the genre classification model.
    """
    print("=" * 65)
    print(f"Starting Music Genre Classification Training [{model_type.upper()}]")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 65)

    # 1. Load Data
    train_loader, val_loader, (n_train, n_val) = get_data_loaders(
        batch_size=batch_size,
        force_recompute=force_recompute
    )
    print(f"Dataset Loaded: {n_train} Train slices | {n_val} Validation slices\n")

    # 2. Initialize Model Architecture
    if model_type.lower() == "crnn":
        model = AudioSpectrogramCRNN(num_classes=NUM_CLASSES)
    else:
        model = AudioSpectrogramCNN(num_classes=NUM_CLASSES)

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: {model.__class__.__name__} ({total_params:,} parameters)")

    # 3. Loss, Optimizer, and LR Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 4. Training Loop with Early Stopping
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = MODELS_DIR / f"best_{model_type.lower()}_model.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        scheduler.step(val_acc)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        # Check for improvement
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "model_type": model_type,
                "num_classes": NUM_CLASSES
            }, best_model_path)
            # Also keep a default symlink / copy as best_model.pth
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            badge = "[BEST]"
        else:
            epochs_no_improve += 1
            badge = ""

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:5.2f}% | "
            f"({elapsed:.1f}s) {badge}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Training Complete in {total_time:.1f}s!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved Best Model Weights to: {best_model_path}")

    # Save History JSON
    history_path = OUTPUTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot & Save Learning Curves
    plot_path = plot_training_history(history, save_path=OUTPUTS_DIR / "training_history.png")
    print(f"Saved Training History Chart to: {plot_path}")
    print("=" * 65)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Genre Classification Model")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "crnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of audio features")
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_recompute=args.recompute
    )
