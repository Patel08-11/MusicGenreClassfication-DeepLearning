"""
Interactive Music Genre Classification Web Application.
Built with Streamlit and PyTorch.
Run using:
    streamlit run app.py
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GENRES,
    NUM_CLASSES,
    SAMPLE_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    RAW_DATA_DIR,
    DEVICE
)
from features.extractor import (
    load_audio,
    compute_mel_spectrogram,
    compute_mfcc,
    compute_chroma,
    extract_audio_summary
)
from features.visualizer import plot_audio_dashboard
from predict import load_trained_model, predict_genre
from train import train_model

# Page Configuration
st.set_page_config(
    page_title="Deep Learning Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header Banner */
    .hero-container {
        padding: 1.8rem;
        background: radial-gradient(circle at top right, rgba(0, 229, 255, 0.12), transparent 70%),
                    linear-gradient(90deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.95));
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
    }

    /* Metric Card */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #00E5FF;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00E5FF;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Confidence Bar */
    .conf-container {
        margin: 0.6rem 0;
    }
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .conf-bar-bg {
        width: 100%;
        background-color: #21262d;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #00E5FF, #7C4DFF);
        transition: width 0.6s ease-in-out;
    }

    /* Report Card */
    .report-box {
        background: #161b22;
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 10px;
        padding: 1.2rem;
        font-family: 'Consolas', 'Courier New', monospace;
        color: #58a6ff;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎵 Deep Learning Music Genre Classifier</div>
    <div class="hero-subtitle">
        Audio feature extraction (Mel-Spectrograms, MFCCs, Chroma) with 2D Convolutional Neural Networks (PyTorch)
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title("⚙️ Model Controls")
model_type = st.sidebar.selectbox("Architecture", ["2D CNN (VGG-style)", "CRNN (CNN + GRU)"])
selected_model_type = "crnn" if "CRNN" in model_type else "cnn"

st.sidebar.markdown(f"**Hardware Acceleration:** `{DEVICE}`")
if torch.cuda.is_available():
    st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Running on CPU")

model_file = MODELS_DIR / "best_model.pth"
model_ready = model_file.exists()
if model_ready:
    st.sidebar.success("✓ Model checkpoint loaded")
else:
    st.sidebar.warning("⚠ No checkpoint found. Run Quickstart or Train model below.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Quick Outputs Folder")
st.sidebar.info(f"`{OUTPUTS_DIR.resolve()}`")


# Navigation Tabs
tab_inference, tab_gallery, tab_comparison, tab_metrics, tab_train = st.tabs([
    "🎧 Classify & Visualize",
    "🖼️ Output PNG & Reports Gallery",
    "🔬 Spectral Comparison",
    "📊 Model Performance",
    "🚀 Train / Retrain"
])


# ==========================================================
# TAB 1: CLASSIFY & VISUALIZE
# ==========================================================
with tab_inference:
    st.markdown("### 1. Select or Upload an Audio Track")
    col_input1, col_input2 = st.columns([1, 1])

    audio_source = col_input1.radio("Input Source:", ["Sample Preset Tracks", "Upload Custom Audio File"], horizontal=True)
    audio_path = None
    target_genre_label = "Unknown"

    if audio_source == "Sample Preset Tracks":
        # Find available demo samples
        sample_files = list(SAMPLES_DIR.glob("*_demo.wav"))
        if not sample_files:
            sample_files = list(RAW_DATA_DIR.glob("*/*.wav"))

        if sample_files:
            sample_names = [f.stem.replace("_demo", "").title() for f in sample_files]
            chosen_name = col_input2.selectbox("Select Genre Sample:", sample_names)
            chosen_idx = sample_names.index(chosen_name)
            audio_path = sample_files[chosen_idx]
            target_genre_label = chosen_name
        else:
            col_input2.warning("No audio samples found. Please generate samples in the Train tab!")
    else:
        uploaded_file = col_input2.file_uploader("Upload audio (.wav, .mp3, .ogg, .flac)", type=["wav", "mp3", "ogg", "flac"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            tfile.write(uploaded_file.read())
            audio_path = Path(tfile.name)
            target_genre_label = Path(uploaded_file.name).stem

    if audio_path and Path(audio_path).exists():
        st.markdown("---")
        st.markdown("### 2. Audio Playback & Classification")
        
        col_play, col_btn = st.columns([2, 1])
        with col_play:
            st.audio(str(audio_path), format="audio/wav")
        with col_btn:
            classify_clicked = st.button("⚡ Classify Music Genre", type="primary", use_container_width=True)

        if classify_clicked:
            if not model_ready:
                st.error("Model checkpoint not found. Please click 'Start Model Training' in the Train tab first!")
            else:
                with st.spinner("Extracting Mel-Spectrogram & running 2D ConvNet inference..."):
                    try:
                        result = predict_genre(
                            audio_path,
                            model_type=selected_model_type,
                            top_k=5,
                            save_spectrogram=True
                        )

                        # Top Genre Banner
                        primary = result["primary_genre"].upper()
                        conf = result["confidence"]
                        st.markdown(f"""
                        <div style="background: rgba(0, 229, 255, 0.1); border: 2px solid #00E5FF; border-radius: 12px; padding: 1.4rem; margin: 1.2rem 0; text-align: center;">
                            <div style="font-size: 0.95rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px;">Predicted Music Genre</div>
                            <div style="font-size: 2.8rem; font-weight: 800; color: #00E5FF; margin: 0.3rem 0;">{primary}</div>
                            <div style="font-size: 1.2rem; color: #e6edf3;">Model Confidence: <b>{conf}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Predictions breakdown
                        st.markdown("#### 🎯 Confidence Ranking Across Genres")
                        for item in result["top_predictions"][:5]:
                            g_name = item["genre"].title()
                            pct = item["percentage"]
                            st.markdown(f"""
                            <div class="conf-container">
                                <div class="conf-label">
                                    <span>{g_name}</span>
                                    <span>{pct}%</span>
                                </div>
                                <div class="conf-bar-bg">
                                    <div class="conf-bar-fill" style="width: {pct}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Acoustic Diagnostics Metrics
                        st.markdown("#### 📊 Acoustic Diagnostics")
                        summary = result["audio_summary"]
                        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                        with dcol1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{summary['estimated_bpm']}</div>
                                <div class="metric-label">Estimated BPM</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with dcol2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{summary['spectral_centroid_hz']} Hz</div>
                                <div class="metric-label">Spectral Centroid</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with dcol3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{summary['zero_crossing_rate']}</div>
                                <div class="metric-label">Zero Crossing Rate</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with dcol4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{summary['duration_sec']}s</div>
                                <div class="metric-label">Track Duration</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Display Dashboard Image Right in the Web Application
                        if result.get("spectrogram_image"):
                            st.markdown("---")
                            st.markdown("#### 🖼️ Output Spectral Representations (Waveform, Mel-Spectrogram, MFCC, Chromagram)")
                            st.caption(f"Generated output file: `{Path(result['spectrogram_image']).name}`")
                            st.image(result["spectrogram_image"], use_container_width=True)

                    except Exception as e:
                        st.error(f"Inference error: {e}")


# ==========================================================
# TAB 2: OUTPUT PNG & REPORTS GALLERY (SHOWS IN WEB APP ONLY)
# ==========================================================
with tab_gallery:
    st.markdown("### 🖼️ Generated Output PNG Images & Evaluation Text Reports")
    st.info("All generated output plots, charts, and classification reports from the project are displayed directly inside this web application view below.")

    # 1. Output PNG Images Section
    png_files = sorted(list(OUTPUTS_DIR.glob("*.png")))
    if png_files:
        st.markdown(f"#### 📸 Visual Output Plots (`.png`) — {len(png_files)} files found")
        
        # Dropdown to inspect any specific output PNG
        selected_png_name = st.selectbox(
            "Select an Output PNG Image to View:",
            [p.name for p in png_files],
            index=0
        )
        selected_png_path = OUTPUTS_DIR / selected_png_name

        if selected_png_path.exists():
            col_img, col_info = st.columns([3, 1])
            with col_img:
                st.image(str(selected_png_path), caption=f"Output: {selected_png_name}", use_container_width=True)
            with col_info:
                st.markdown(f"**Filename:** `{selected_png_name}`")
                st.markdown(f"**Size:** `{round(selected_png_path.stat().st_size / 1024, 1)} KB`")
                with open(selected_png_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PNG",
                        data=f.read(),
                        file_name=selected_png_name,
                        mime="image/png",
                        use_container_width=True
                    )

        # Thumbnail Grid of all PNG files
        st.markdown("---")
        st.markdown("#### 🖼️ All Output Plots at a Glance")
        cols = st.columns(min(len(png_files), 3))
        for idx, png_file in enumerate(png_files):
            with cols[idx % 3]:
                st.image(str(png_file), caption=png_file.name, use_container_width=True)
    else:
        st.warning("No PNG outputs found in outputs/ yet. Run a prediction or training to generate plots!")

    # 2. Output Text Reports Section
    st.markdown("---")
    st.markdown("#### 📄 Evaluation Text Report (`evaluation_report.txt`)")
    report_file = OUTPUTS_DIR / "evaluation_report.txt"
    if report_file.exists():
        with open(report_file, "r") as f:
            report_text = f.read()

        col_rep1, col_rep2 = st.columns([3, 1])
        with col_rep1:
            st.code(report_text, language="text")
        with col_rep2:
            st.markdown(f"**File:** `evaluation_report.txt`")
            st.markdown(f"**Size:** `{report_file.stat().st_size} bytes`")
            st.download_button(
                label="📥 Download Text Report",
                data=report_text,
                file_name="evaluation_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Run `python evaluate.py` to generate the evaluation report text file.")

    # 3. Output Training JSON History
    history_file = OUTPUTS_DIR / "training_history.json"
    if history_file.exists():
        st.markdown("---")
        st.markdown("#### 📈 Training History Data (`training_history.json`)")
        with open(history_file, "r") as f:
            history_data = json.load(f)
        
        with st.expander("🔍 View Raw Training History JSON"):
            st.json(history_data)


# ==========================================================
# TAB 3: SPECTRAL COMPARISON
# ==========================================================
with tab_comparison:
    st.markdown("### 🔬 Compare Audio Spectrograms Across Genres")
    st.info("Spectrograms visually encode the musical DNA of genres. Notice how metal shows dense high-frequency saturation (distortion), classical displays clean sustained horizontal harmonics, and disco/hiphop exhibit strong vertical transient lines from drum hits!")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        genre1 = st.selectbox("Genre A:", GENRES, index=1)  # classical
    with col_g2:
        genre2 = st.selectbox("Genre B:", GENRES, index=6)  # metal

    sample1 = SAMPLES_DIR / f"{genre1}_demo.wav"
    sample2 = SAMPLES_DIR / f"{genre2}_demo.wav"

    if sample1.exists() and sample2.exists():
        y1, sr1 = load_audio(sample1, sr=SAMPLE_RATE, duration=6.0)
        y2, sr2 = load_audio(sample2, sr=SAMPLE_RATE, duration=6.0)

        mel1 = compute_mel_spectrogram(y1, sr=sr1, normalize=False)
        mel2 = compute_mel_spectrogram(y2, sr=sr2, normalize=False)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"#### {genre1.title()} Spectrogram")
            st.audio(str(sample1))
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            fig1.patch.set_facecolor("#0F1117")
            ax1.set_facecolor("#161923")
            import librosa.display
            librosa.display.specshow(mel1, sr=sr1, hop_length=512, x_axis="time", y_axis="mel", ax=ax1, cmap="magma")
            ax1.set_title(f"{genre1.title()} Mel-Spectrogram", color="#FFFFFF", fontweight="bold")
            st.pyplot(fig1)
            plt.close(fig1)

        with col_p2:
            st.markdown(f"#### {genre2.title()} Spectrogram")
            st.audio(str(sample2))
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            fig2.patch.set_facecolor("#0F1117")
            ax2.set_facecolor("#161923")
            librosa.display.specshow(mel2, sr=sr2, hop_length=512, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
            ax2.set_title(f"{genre2.title()} Mel-Spectrogram", color="#FFFFFF", fontweight="bold")
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.warning("Generate sample audio first in the Train tab to explore side-by-side comparisons!")


# ==========================================================
# TAB 4: MODEL PERFORMANCE
# ==========================================================
with tab_metrics:
    st.markdown("### 📊 Model Architecture & Training Evaluation")
    
    col_hist, col_cm = st.columns(2)
    history_img = OUTPUTS_DIR / "training_history.png"
    cm_img = OUTPUTS_DIR / "confusion_matrix.png"

    with col_hist:
        st.markdown("#### Training & Validation Curves")
        if history_img.exists():
            st.image(str(history_img), use_container_width=True)
        else:
            st.info("Train the model to view loss and accuracy curves.")

    with col_cm:
        st.markdown("#### Confusion Matrix Heatmap")
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)
        else:
            st.info("Run evaluation to view the Confusion Matrix.")

    if report_file.exists():
        st.markdown("#### Detailed Per-Genre Classification Report")
        with open(report_file, "r") as f:
            st.code(f.read(), language="text")


# ==========================================================
# TAB 5: TRAIN / RETRAIN
# ==========================================================
with tab_train:
    st.markdown("### 🚀 Live Model Training & Dataset Management")
    st.write("Train or retrain your Deep Learning model on audio Mel-Spectrograms directly from this dashboard.")

    col_t1, col_t2, col_t3 = st.columns(3)
    t_epochs = col_t1.slider("Epochs", min_value=5, max_value=50, value=15, step=5)
    t_batch = col_t2.selectbox("Batch Size", [16, 32, 64], index=1)
    t_lr = col_t3.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0)

    col_run_train, col_run_samples = st.columns(2)
    with col_run_samples:
        if st.button("🎹 Synthesize Demo Audio Samples", use_container_width=True):
            with st.spinner("Synthesizing audio samples across all 10 genres..."):
                from dataset.generate_samples import generate_all_samples
                generate_all_samples(tracks_per_genre=6, duration=15.0)
                st.success("Successfully generated audio dataset in data/raw/!")

    with col_run_train:
        if st.button("🔥 Start Model Training", type="primary", use_container_width=True):
            status_box = st.empty()
            status_box.info("Training started... checking GPU and loading dataset...")
            try:
                hist = train_model(
                    model_type=selected_model_type,
                    epochs=t_epochs,
                    batch_size=t_batch,
                    lr=t_lr,
                    force_recompute=True
                )
                status_box.success(f"✓ Training finished! Best validation accuracy: {max(hist['val_acc'])}%")
                st.rerun()
            except Exception as e:
                status_box.error(f"Training error: {e}")
