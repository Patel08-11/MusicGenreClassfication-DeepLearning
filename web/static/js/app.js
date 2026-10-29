/**
 * Main Application Controller for SONIX AI Studio Workstation
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Visualizer Engine
    const visualizer = new AudioVisualizerEngine('audio-player', 'spectrum-canvas');

    // 2. Audio & Transport Elements
    const audio = document.getElementById('audio-player');
    const btnPlay = document.getElementById('btn-play');
    const btnStop = document.getElementById('btn-stop');
    const playIcon = document.getElementById('play-icon');
    const pauseIcon = document.getElementById('pause-icon');
    const seekBar = document.getElementById('seek-bar');
    const timeCurrent = document.getElementById('time-current');
    const timeDuration = document.getElementById('time-duration');
    const volumeSlider = document.getElementById('volume-slider');
    const btnClassify = document.getElementById('btn-run-inference');

    // Display elements
    const trackNameDisplay = document.getElementById('current-track-name');
    const vinylGenreLabel = document.getElementById('vinyl-genre-label');
    const holoGenreTitle = document.getElementById('holo-genre-title');
    const holoConfidenceVal = document.getElementById('holo-confidence-val');
    const holoGenreDesc = document.getElementById('holo-genre-desc');
    const genreBarsContainer = document.getElementById('genre-bars-container');
    const specTrackLabel = document.getElementById('spec-track-label');
    const liveSpectrogramImg = document.getElementById('live-spectrogram-img');

    // Telemetry elements
    const teleBpm = document.getElementById('tele-bpm');
    const teleCentroid = document.getElementById('tele-centroid');
    const teleRolloff = document.getElementById('tele-rolloff');
    const teleZcr = document.getElementById('tele-zcr');

    let currentTrackId = 'metal';
    let availableTracks = [];

    // Genre Mood Descriptions Dictionary
    const GENRE_DESCRIPTIONS = {
        metal: "Heavy harmonic distortion, intense double-kick transients, and overdriven power chords.",
        classical: "Rich orchestral dynamics, sustained melodic strings, and acoustic harmonic cadence.",
        blues: "Expressive 12-bar blues shuffle rhythm with characteristic minor pentatonic guitar riffs.",
        disco: "Four-on-the-floor driving kick drum, syncopated octave bassline, and upbeat sizzling hats.",
        hiphop: "Deep 808 sub-bass pressure, punchy rhythmic snares, and fast hi-hat roll textures.",
        jazz: "Intricate 7th chord voicings, syncopated walking acoustic bass, and swing ride cymbals.",
        rock: "Solid backbeat drums on beats 2 & 4, saturated rhythm guitars, and melodic bass drive.",
        pop: "High-sparkle vocal harmonics, bright synth chord hooks, and clean 4-beat electro pulse.",
        reggae: "Classic one-drop bass on beat 3, relaxed dub tempo, and off-beat guitar skank chop.",
        country: "Boom-chick acoustic guitar strumming, clear vocal warmth, and rustic major triads."
    };

    // ==========================================================
    // 3. Tab Switching Navigation
    // ==========================================================
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabPanels = document.querySelectorAll('.tab-panel');

    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;
            navTabs.forEach(t => t.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) targetPanel.classList.add('active');

            if (targetId === 'tab-metrics' || targetId === 'tab-gallery') {
                loadMetricsAndGallery();
            }
        });
    });

    // ==========================================================
    // 4. Audio Playback & Transport Controls
    // ==========================================================
    btnPlay.addEventListener('click', () => {
        if (audio.paused) {
            audio.play().then(() => updatePlayIcon(true)).catch(e => console.error(e));
        } else {
            audio.pause();
            updatePlayIcon(false);
        }
    });

    btnStop.addEventListener('click', () => {
        audio.pause();
        audio.currentTime = 0;
        updatePlayIcon(false);
    });

    function updatePlayIcon(isPlaying) {
        if (isPlaying) {
            playIcon.classList.add('hidden');
            pauseIcon.classList.remove('hidden');
        } else {
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
        }
    }

    audio.addEventListener('timeupdate', () => {
        if (!isNaN(audio.duration) && audio.duration > 0) {
            const pct = (audio.currentTime / audio.duration) * 100;
            seekBar.value = pct;
            timeCurrent.textContent = formatTime(audio.currentTime);
            timeDuration.textContent = formatTime(audio.duration);
        }
    });

    seekBar.addEventListener('input', () => {
        if (!isNaN(audio.duration)) {
            audio.currentTime = (seekBar.value / 100) * audio.duration;
        }
    });

    volumeSlider.addEventListener('input', (e) => {
        audio.volume = parseFloat(e.target.value);
    });

    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }

    // ==========================================================
    // 5. Load Tracks & Initialize Sound Deck
    // ==========================================================
    async function loadTracks() {
        try {
            const res = await fetch('/api/tracks');
            const data = await res.json();
            availableTracks = data.tracks;

            renderGenreEqualizerBars({});
            renderPresetCards(availableTracks);

            // Set default track
            selectTrack('metal');
        } catch (e) {
            console.error("Failed to load tracks:", e);
        }
    }

    function renderPresetCards(tracks) {
        const grid = document.getElementById('preset-tracks-grid');
        if (!grid) return;
        grid.innerHTML = '';

        tracks.forEach(track => {
            const tile = document.createElement('div');
            tile.className = `track-tile ${track.id === currentTrackId ? 'active-selected' : ''}`;
            tile.id = `tile-${track.id}`;

            tile.innerHTML = `
                <div class="tile-info">
                    <span class="tile-genre-badge">${track.genre}</span>
                    <h4 class="tile-title">${track.title}</h4>
                    <p class="tile-desc">${track.bpm} BPM // ${track.mood}</p>
                </div>
                <div class="tile-actions">
                    <button class="tile-play-btn" data-id="${track.id}">▶ Play</button>
                    <button class="tile-analyze-btn" data-id="${track.id}">⚡ Analyze</button>
                </div>
            `;

            tile.querySelector('.tile-play-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                selectTrack(track.id, true);
            });

            tile.querySelector('.tile-analyze-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                selectTrack(track.id, false);
                runClassification(track.id);
            });

            tile.addEventListener('click', () => {
                selectTrack(track.id, false);
            });

            grid.appendChild(tile);
        });
    }

    function selectTrack(trackId, autoPlay = false) {
        currentTrackId = trackId;
        const track = availableTracks.find(t => t.id === trackId);
        if (!track) return;

        // Update UI displays
        trackNameDisplay.textContent = track.title;
        vinylGenreLabel.textContent = track.genre.toUpperCase();
        if (specTrackLabel) specTrackLabel.textContent = `Track: ${track.id}_demo.wav`;

        // Update tile styles
        document.querySelectorAll('.track-tile').forEach(t => t.classList.remove('active-selected'));
        const activeTile = document.getElementById(`tile-${trackId}`);
        if (activeTile) activeTile.classList.add('active-selected');

        // Set audio source
        audio.src = `/api/audio/${trackId}`;
        audio.load();

        if (autoPlay) {
            audio.play().then(() => updatePlayIcon(true)).catch(e => console.error(e));
        }

        // Update Spectrogram preview
        if (liveSpectrogramImg) {
            liveSpectrogramImg.src = `/api/spectrogram/${trackId}?t=${Date.now()}`;
        }
    }

    // ==========================================================
    // 6. 10-Genre Real-Time Equalizer Bar Graph
    // ==========================================================
    function renderGenreEqualizerBars(probabilities = {}, topGenre = '') {
        if (!genreBarsContainer) return;
        const genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"];

        genreBarsContainer.innerHTML = '';
        genres.forEach(genre => {
            const pct = probabilities[genre] !== undefined ? probabilities[genre] : 0;
            const isTop = genre === topGenre;

            const row = document.createElement('div');
            row.className = 'genre-bar-item';
            row.innerHTML = `
                <span class="gbar-name">${genre}</span>
                <div class="gbar-track">
                    <div class="gbar-fill ${isTop ? 'top-match' : ''}" style="width: ${pct}%;"></div>
                </div>
                <span class="gbar-pct">${pct.toFixed(1)}%</span>
            `;
            genreBarsContainer.appendChild(row);
        });
    }

    // ==========================================================
    // 7. AI Genre Inference Execution
    // ==========================================================
    btnClassify.addEventListener('click', () => {
        runClassification(currentTrackId);
    });

    async function runClassification(trackId, uploadedFile = null) {
        btnClassify.disabled = true;
        btnClassify.innerHTML = '<span>⚡ ANALYZING...</span>';

        const formData = new FormData();
        if (uploadedFile) {
            formData.append('audio_file', uploadedFile);
        } else {
            formData.append('track_id', trackId);
        }

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error("Prediction API failed");
            const result = await res.json();

            // Update Master Holographic Badge
            const primary = result.primary_genre.toUpperCase();
            holoGenreTitle.textContent = primary;
            holoConfidenceVal.textContent = `${result.confidence}%`;
            holoGenreDesc.textContent = GENRE_DESCRIPTIONS[result.primary_genre] || "Acoustic audio signature analyzed.";

            // Animate 10-Genre Bars
            renderGenreEqualizerBars(result.all_probabilities, result.primary_genre);

            // Update Telemetry Dials
            const summary = result.audio_summary;
            teleBpm.textContent = `${summary.estimated_bpm}`;
            teleCentroid.textContent = `${summary.spectral_centroid_hz} Hz`;
            teleRolloff.textContent = `${summary.spectral_rolloff_hz} Hz`;
            teleZcr.textContent = `${summary.zero_crossing_rate}`;

            // Update Spectral Image
            if (result.spectrogram_image_url && liveSpectrogramImg) {
                liveSpectrogramImg.src = `${result.spectrogram_image_url}?t=${Date.now()}`;
            }

            // Scroll to view if on mobile/small screen
            console.log("Prediction complete:", result);
        } catch (e) {
            console.error("Classification error:", e);
            alert("Error running genre prediction: " + e.message);
        } finally {
            btnClassify.disabled = false;
            btnClassify.innerHTML = '<span>⚡ CLASSIFY GENRE</span>';
        }
    }

    // ==========================================================
    // 8. Drag and Drop Audio File Upload
    // ==========================================================
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse-file');
    const uploadStatus = document.getElementById('upload-status');

    if (dropZone && fileInput) {
        btnBrowse.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('click', () => fileInput.click());

        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) handleUploadedAudio(files[0]);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) handleUploadedAudio(fileInput.files[0]);
        });
    }

    function handleUploadedAudio(file) {
        uploadStatus.innerHTML = `<span style="color: var(--neon-cyan);">Analyzing uploaded audio: <b>${file.name}</b>...</span>`;

        // Load into audio player for listening
        const objectUrl = URL.createObjectURL(file);
        audio.src = objectUrl;
        trackNameDisplay.textContent = file.name;
        vinylGenreLabel.textContent = "CUSTOM";

        // Switch to Studio tab and run classification
        document.querySelector('[data-tab="tab-player"]').click();
        runClassification(null, file);
    }

    // ==========================================================
    // 9. Metrics & Output Gallery Loader
    // ==========================================================
    async function loadMetricsAndGallery() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();

            // Classification Report text
            const reportEl = document.getElementById('classification-report-text');
            if (reportEl) reportEl.textContent = data.report;

            // Output Gallery Grid
            const gallery = document.getElementById('gallery-container');
            if (gallery && data.all_pngs) {
                gallery.innerHTML = '';
                data.all_pngs.forEach(pngName => {
                    const card = document.createElement('div');
                    card.className = 'gallery-card';
                    card.innerHTML = `
                        <img src="/api/outputs/${pngName}" alt="${pngName}" class="gallery-thumb">
                        <div class="gallery-meta">
                            <span class="gallery-title">${pngName}</span>
                            <a href="/api/outputs/${pngName}" download="${pngName}" class="gallery-dl-btn">📥 Download</a>
                        </div>
                    `;
                    gallery.appendChild(card);
                });
            }
        } catch (e) {
            console.error("Failed to load metrics/gallery:", e);
        }
    }

    // Initialize initial track catalog
    loadTracks();
});
