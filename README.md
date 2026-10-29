# 🎵 Deep Learning Music Genre Classification (Easy-Way)

An end-to-end, production-ready, accessible Deep Learning project that extracts audio spectral features (**Mel-Spectrograms, STFT, MFCCs, Chromagrams**) and trains **2D Convolutional Neural Networks (CNN)** and **CRNNs** in **PyTorch** to classify music tracks into genres.

![Music Genre Classification Pipeline](outputs/audio_spectral_dashboard.png)

---

## 🌟 Highlights

- **Zero-Friction "Easy-Way" Setup**: Includes a built-in synthetic multi-genre audio generator so you can test, train, and run the complete deep learning pipeline immediately without waiting for huge dataset downloads.
- **Full GTZAN Support**: Automated 1-click downloader and preprocessor for the official 1.2 GB benchmark GTZAN dataset (1,000 tracks across 10 genres).
- **Audio Feature Extraction Pipeline**:
  - **Mel-Scale Spectrograms** (Log-power dB scaled, 128 Mel bands)
  - **MFCCs** (Mel-Frequency Cepstral Coefficients for timbral analysis)
  - **Chromagrams** (12 pitch classes for harmonic analysis)
  - **Acoustic Diagnostics** (Tempo/BPM, Spectral Centroid brightness, Rolloff, Zero Crossing Rate)
- **Deep Learning Architectures**:
  - `AudioSpectrogramCNN`: 4-block 2D Convolutional Neural Network with Batch Normalization, Dropout, and Adaptive Pooling.
  - `AudioSpectrogramCRNN`: Combines 2D CNN (spatial/spectral timbre) with Bidirectional GRU (rhythmic temporal sequence).
- **GPU Acceleration**: Native PyTorch CUDA support (tested on NVIDIA GeForce RTX 4050 GPU).
- **Interactive Web App**: Modern **Streamlit** dashboard with audio playback, real-time genre predictions, animated confidence bars, side-by-side genre spectrogram comparisons, and live training controls.

---

## 📂 Project Directory Structure

```text
dl/
├── config.py                 # Central configurations (sampling rate, FFT, genres, hyperparameters)
├── quickstart.py             # 1-click end-to-end runner (generates audio -> extracts -> trains -> evaluates -> predicts)
├── train.py                  # PyTorch training pipeline with learning rate scheduler & early stopping
├── evaluate.py               # Evaluation script: Top-1/Top-3 accuracy, classification report & confusion matrix
├── predict.py                # Command-line inference on any custom audio file with confidence breakdown
├── app.py                    # Interactive Streamlit Web Application
├── requirements.txt          # Python dependencies list
├── README.md                 # Documentation & theory guide
│
├── dataset/
│   ├── audio_loader.py       # Audio slicing, caching (.npz), and PyTorch DataLoader
│   ├── generate_samples.py   # Multi-genre synthetic audio synthesizer (instant testing)
│   └── download_dataset.py   # GTZAN dataset automated downloader and extractor
│
├── features/
│   ├── extractor.py          # Mel-spectrogram, MFCC, STFT, Chroma, and summary extraction
│   └── visualizer.py         # Matplotlib spectral dashboards, training curves, confusion matrix
│
├── models/
│   ├── cnn_classifier.py     # 2D Spectrogram Convolutional Neural Network
│   └── crnn_classifier.py    # Convolutional Recurrent Neural Network (CNN + Bi-GRU)
│
├── data/
│   ├── raw/                  # Audio tracks organized by genre folder (e.g. data/raw/rock/*.wav)
│   ├── processed/            # Cached Mel-spectrogram tensors (.npz) for fast training
│   └── samples/              # Quick test audio clips
│
├── saved_models/             # Saved PyTorch model weights (best_model.pth)
└── outputs/                  # Exported plots (confusion matrix, loss curves, spectrograms)
```

---

## ⚡ 1-Minute Quickstart

Run the entire pipeline (dataset verification, feature extraction, model training, evaluation, and test prediction) with a single command:

```bash
python quickstart.py
```

This will:
1. Verify audio samples in `data/raw/` (or synthesize them automatically).
2. Segment tracks into 3-second slices and compute $(1, 128, 130)$ Mel-spectrograms.
3. Train the `AudioSpectrogramCNN` with GPU acceleration.
4. Save the best model weights to `saved_models/best_model.pth`.
5. Generate the Confusion Matrix and Training History curves in `outputs/`.
6. Run inference on sample tracks and print the top predicted genres with confidence bars!

---

## 🌐 Launch the Interactive Web Application

Start the modern web dashboard to listen to audio, classify custom tracks, and inspect spectrograms in real-time:

```bash
streamlit run app.py
```

The web app lets you:
- **Upload any MP3 or WAV file** or pick from built-in sample tracks.
- **Listen to the track** with an embedded audio player.
- **Classify with 1 click** and view animated probability bars for all 10 genres.
- **Inspect the 4-panel spectral dashboard** (Waveform, Mel-Spectrogram, MFCC, Chromagram).
- **Compare spectrograms of 2 genres side by side** (e.g., Classical vs Metal).
- **Train or retrain** your model directly with custom epochs and learning rate.

---

## 🧠 Deep Learning & Audio Theory: How It Works

### 1. From Raw Sound to Visual Spectrograms
Audio is recorded as a 1D time-domain waveform representing air pressure variations over time. However, 1D waveforms make it difficult for deep neural networks to distinguish musical chords and timbres.

1. **Short-Time Fourier Transform (STFT)**: Slices the audio into overlapping time windows ($N_{FFT} = 2048$, hop size $= 512$) and calculates the Fast Fourier Transform (FFT) for each window, converting sound from the **Time Domain** to the **Frequency Domain**.
2. **The Mel Scale**: The human ear perceives pitch logarithmically (it is much easier to distinguish 200 Hz from 300 Hz than 10,000 Hz from 10,100 Hz). The Mel scale warps standard Hertz into human perceptual frequency:
   $$m = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$
3. **Decibel (dB) Scaling**: Converts power to a logarithmic scale ($10 \log_{10}(P / P_{ref})$), matching human perception of loudness.
4. **Result**: A 2D matrix of shape $(128, 130)$ where the Y-axis is frequency, the X-axis is time, and the pixel intensity is energy in dB.

### 2. Why 2D CNNs for Audio?
By converting audio into a Mel-spectrogram, we can treat sound classification as an **acoustic image recognition** task:
- **Horizontal lines** represent sustained musical notes and harmonics (prominent in *Classical* and *Jazz*).
- **Vertical lines / spikes** represent sharp percussive transients like drum kicks and snares (prominent in *Hiphop*, *Disco*, and *Rock*).
- **Dense high-frequency bands** represent harmonic distortion and overdrive (prominent in *Metal*).

The 2D Convolutional filters slide across both the frequency and time axes to learn these hierarchical acoustic patterns automatically!

---

## 🛠️ CLI Usage Guide

### 1. Predict Genre of Any Custom Audio File
```bash
python predict.py --audio path/to/your_song.wav
```
Options:
- `--audio <path>`: Path to audio file (`.wav`, `.mp3`, `.ogg`, `.flac`)
- `--model <cnn|crnn>`: Model architecture (default: `cnn`)
- `--top-k <num>`: Number of top genres to display (default: 5)

### 2. Train the Model
```bash
python train.py --model cnn --epochs 25 --batch-size 32 --lr 0.001
```
Options:
- `--model`: `cnn` or `crnn`
- `--epochs`: Number of training epochs (default: 25)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--recompute`: Force re-extraction of Mel-spectrograms from raw audio

### 3. Evaluate Model on Test Data
```bash
python evaluate.py --model cnn --checkpoint saved_models/best_model.pth
```
Outputs:
- Top-1 and Top-3 accuracy
- Per-genre Precision, Recall, and F1-score
- Normalized Confusion Matrix plot saved to `outputs/confusion_matrix.png`

### 4. Download Full GTZAN Dataset (Optional)
```bash
python dataset/download_dataset.py
```
Downloads and extracts the official benchmark dataset containing 1,000 tracks across 10 genres.

---

## 🎓 Viva / Interview Q&A for Deep Learning Projects

**Q1: Why do we use Mel-Spectrograms instead of raw audio waveforms?**
> *Raw audio has high temporal dimensionality (e.g. 22,050 samples per second) without explicit frequency structure. Mel-spectrograms compress this data into frequency vs. time representations aligned with human hearing, allowing standard 2D ConvNets to detect timbral and harmonic features.*

**Q2: Why slice audio into 3-second segments?**
> *A 30-second GTZAN track yields ten 3-second slices. This data augmentation increases the training set by 10x (from 1,000 tracks to 10,000 slices) while keeping the input tensor compact $(1, 128, 130)$, enabling fast GPU training and higher generalization.*

**Q3: How is a full song classified during inference?**
> *The track is sliced into 3-second segments, each segment is passed through the CNN, and the resulting softmax probability vectors are averaged across all segments (soft voting ensemble over time).*

---

## 📜 License
MIT License. Open for educational and research use.
