/**
 * Web Audio API Engine & Real-Time Spectrum Visualizer
 * Drives the real-time frequency canvas, dual analog LED VU meters,
 * and vinyl turntable synchronization.
 */

class AudioVisualizerEngine {
    constructor(audioElementId, canvasId) {
        this.audio = document.getElementById(audioElementId);
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        this.audioContext = null;
        this.analyser = null;
        this.source = null;
        this.dataArray = null;
        this.animationId = null;
        this.isInitialized = false;

        // VU meter LED elements
        this.leftLeds = Array.from(document.querySelectorAll('#vu-left .led')).reverse(); // bottom to top
        this.rightLeds = Array.from(document.querySelectorAll('#vu-right .led')).reverse();

        this.setupEventListeners();
    }

    initAudioContext() {
        if (this.isInitialized) return;
        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContextClass();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.82;

            this.source = this.audioContext.createMediaElementSource(this.audio);
            this.source.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);

            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            this.isInitialized = true;
        } catch (e) {
            console.warn("Web Audio API not supported or autoplay restriction:", e);
        }
    }

    setupEventListeners() {
        if (!this.audio) return;

        this.audio.addEventListener('play', () => {
            this.initAudioContext();
            if (this.audioContext && this.audioContext.state === 'suspended') {
                this.audioContext.resume();
            }
            this.startVisualizer();
            this.setTurntableState(true);
        });

        this.audio.addEventListener('pause', () => {
            this.stopVisualizer();
            this.setTurntableState(false);
            this.clearVuMeters();
        });

        this.audio.addEventListener('ended', () => {
            this.stopVisualizer();
            this.setTurntableState(false);
            this.clearVuMeters();
        });
    }

    setTurntableState(isPlaying) {
        const disc = document.getElementById('vinyl-disc');
        const tonearm = document.getElementById('tonearm');
        if (disc) {
            if (isPlaying) disc.classList.add('spinning');
            else disc.classList.remove('spinning');
        }
        if (tonearm) {
            if (isPlaying) tonearm.classList.add('playing');
            else tonearm.classList.remove('playing');
        }
    }

    startVisualizer() {
        if (!this.ctx || !this.analyser) return;

        const renderFrame = () => {
            this.animationId = requestAnimationFrame(renderFrame);
            this.analyser.getByteFrequencyData(this.dataArray);

            const width = this.canvas.width;
            const height = this.canvas.height;

            // Clear with dark obsidian background
            this.ctx.fillStyle = '#05070B';
            this.ctx.fillRect(0, 0, width, height);

            // Draw subtle horizontal grid lines
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
            this.ctx.lineWidth = 1;
            for (let y = 15; y < height; y += 20) {
                this.ctx.beginPath();
                this.ctx.moveTo(0, y);
                this.ctx.lineTo(width, y);
                this.ctx.stroke();
            }

            // Draw Glowing Frequency Spectrum Bars
            const numBars = 48;
            const barWidth = (width / numBars) - 2;
            let totalVolume = 0;

            for (let i = 0; i < numBars; i++) {
                const dataIndex = Math.floor(i * (this.dataArray.length / numBars));
                const value = this.dataArray[dataIndex] || 0;
                totalVolume += value;
                const percent = value / 255;
                const barHeight = Math.max(3, percent * (height - 12));

                const x = i * (barWidth + 2) + 2;
                const y = height - barHeight;

                // Gradient from Cyan to Electric Purple
                const grad = this.ctx.createLinearGradient(0, height, 0, y);
                grad.addColorStop(0, '#00F0FF');
                grad.addColorStop(0.7, '#8A2BE2');
                grad.addColorStop(1, '#FF007F');

                this.ctx.fillStyle = grad;
                this.ctx.fillRect(x, y, barWidth, barHeight);

                // Peak cap dot
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.fillRect(x, Math.max(0, y - 2), barWidth, 2);
            }

            // Update Analog VU Meters
            const avgVolume = (totalVolume / numBars) / 255; // 0.0 to 1.0
            this.updateVuMeters(avgVolume);
        };

        renderFrame();
    }

    stopVisualizer() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        if (this.ctx) {
            this.ctx.fillStyle = '#05070B';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    updateVuMeters(volume) {
        const totalLeds = this.leftLeds.length; // 10 leds
        // Add subtle stereo variance
        const leftLevel = Math.min(1.0, volume * (1.1 + Math.random() * 0.15));
        const rightLevel = Math.min(1.0, volume * (1.05 + Math.random() * 0.15));

        const activeLeft = Math.floor(leftLevel * totalLeds);
        const activeRight = Math.floor(rightLevel * totalLeds);

        this.leftLeds.forEach((led, idx) => {
            if (idx <= activeLeft && activeLeft > 0) led.classList.add('on');
            else led.classList.remove('on');
        });

        this.rightLeds.forEach((led, idx) => {
            if (idx <= activeRight && activeRight > 0) led.classList.add('on');
            else led.classList.remove('on');
        });
    }

    clearVuMeters() {
        this.leftLeds.forEach(l => l.classList.remove('on'));
        this.rightLeds.forEach(l => l.classList.remove('on'));
    }
}
