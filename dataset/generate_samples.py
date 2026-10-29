"""
Synthetic Audio Sample Generator for Music Genre Classification.
Synthesizes distinctive genre audio tracks using characteristic musical textures:
- Tempo (BPM), beat accents, harmonic progressions, drum patterns, and distortion.
Allows the user to immediately train and test the full Deep Learning pipeline
without waiting for massive dataset downloads!
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import soundfile as sf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    SAMPLES_DIR,
    GENRES,
    SAMPLE_RATE
)


def synthesize_tone(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a pure tone with soft attack and release envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.ones_like(t)
    fade_len = int(min(sr * 0.01, len(t) // 4))
    if fade_len > 0:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    return np.sin(2 * np.pi * freq * t) * envelope


def synthesize_kick(sr: int = SAMPLE_RATE, duration: float = 0.25) -> np.ndarray:
    """Synthesize a deep kick drum sound (pitch drop + decay)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 150.0 * np.exp(-t * 25) + 45.0
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 15)
    return (np.sin(phase) * env).astype(np.float32)


def synthesize_snare(sr: int = SAMPLE_RATE, duration: float = 0.2) -> np.ndarray:
    """Synthesize a snare drum sound (noise burst + tone)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.uniform(-1, 1, len(t))
    tone = np.sin(2 * np.pi * 180 * t)
    env = np.exp(-t * 22)
    return ((0.75 * noise + 0.25 * tone) * env).astype(np.float32)


def synthesize_hihat(sr: int = SAMPLE_RATE, duration: float = 0.08, open_hat: bool = False) -> np.ndarray:
    """Synthesize a metallic hi-hat (high-pass noise)."""
    dur = duration if not open_hat else duration * 3
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.uniform(-1, 1, len(t))
    # Simple high-pass effect by differencing
    noise = np.diff(noise, prepend=0)
    decay = 40 if not open_hat else 15
    env = np.exp(-t * decay)
    return (noise * env).astype(np.float32)


def overlay(target: np.ndarray, sound: np.ndarray, start_idx: int, gain: float = 1.0) -> None:
    """Overlay sound into target array at start_idx with gain."""
    if start_idx >= len(target):
        return
    end_idx = min(start_idx + len(sound), len(target))
    sound_len = end_idx - start_idx
    target[start_idx:end_idx] += sound[:sound_len] * gain


def generate_genre_audio(genre: str, duration: float = 12.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate a musically distinctive track for the given genre.
    """
    total_samples = int(sr * duration)
    track = np.zeros(total_samples, dtype=np.float32)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    if genre == "blues":
        # 12-bar blues shuffle, minor pentatonic notes (E, G, A, Bb, B, D), 105 BPM
        bpm = 105
        beat_dur = 60.0 / bpm
        notes = [82.4, 98.0, 110.0, 116.5, 123.5, 146.8]  # Low blues scale
        step = beat_dur / 2.0
        cur_time = 0.0
        while cur_time < duration:
            note = np.random.choice(notes)
            sound = synthesize_tone(note, step * 0.9, sr)
            overlay(track, sound, int(cur_time * sr), gain=0.6)
            if int(cur_time / beat_dur) % 2 == 1:
                overlay(track, synthesize_snare(sr), int(cur_time * sr), gain=0.3)
            cur_time += step

    elif genre == "classical":
        # Gentle arpeggios, sustained strings/chords (A minor, D minor, F major, E major), slow tempo
        chords = [
            [220.0, 261.6, 329.6],  # Am
            [174.6, 220.0, 261.6],  # F
            [146.8, 174.6, 220.0],  # Dm
            [164.8, 207.6, 246.9],  # E
        ]
        chord_dur = 2.5
        for i, c_idx in enumerate(range(int(duration / chord_dur) + 1)):
            chord = chords[i % len(chords)]
            start_t = i * chord_dur
            for freq in chord:
                # Sustained cello/piano tones with overtones
                tone = synthesize_tone(freq, chord_dur * 1.2, sr) * 0.3
                tone += synthesize_tone(freq * 2, chord_dur * 1.2, sr) * 0.15
                tone += synthesize_tone(freq * 3, chord_dur * 1.2, sr) * 0.05
                overlay(track, tone, int(start_t * sr), gain=0.5)

    elif genre == "metal":
        # Fast double kick (150 BPM), distorted guitar power chords with heavy clipping
        bpm = 150
        beat_dur = 60.0 / bpm
        sixteenth = beat_dur / 4.0
        # Fast double-kick
        cur_t = 0.0
        while cur_t < duration:
            overlay(track, synthesize_kick(sr, 0.15), int(cur_t * sr), gain=0.5)
            overlay(track, synthesize_hihat(sr, 0.04), int(cur_t * sr), gain=0.2)
            cur_t += sixteenth * 2

        # Power chords with heavy saturation / distortion
        power_chords = [82.4, 73.4, 98.0, 87.3]  # E, D, G, F
        cur_t = 0.0
        while cur_t < duration:
            root = np.random.choice(power_chords)
            fifth = root * 1.4983  # Perfect 5th
            octave = root * 2.0
            riff = synthesize_tone(root, beat_dur, sr) + synthesize_tone(fifth, beat_dur, sr) + synthesize_tone(octave, beat_dur, sr)
            # Hard clipping distortion
            riff = np.tanh(riff * 4.0) * 0.6
            overlay(track, riff, int(cur_t * sr), gain=0.6)
            cur_t += beat_dur

    elif genre == "disco":
        # 4-on-the-floor kick, 124 BPM, disco octave bass, upbeat open hi-hat
        bpm = 124
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        while cur_t < duration:
            # 4-on-the-floor kick
            overlay(track, synthesize_kick(sr), int(cur_t * sr), gain=0.7)
            # Snare on 2 and 4
            beat_num = int(round(cur_t / beat_dur)) % 4
            if beat_num in (1, 3):
                overlay(track, synthesize_snare(sr), int(cur_t * sr), gain=0.5)
            # Off-beat hi-hat (sizzle)
            overlay(track, synthesize_hihat(sr, 0.08, open_hat=True), int((cur_t + beat_dur / 2.0) * sr), gain=0.4)
            # Bass octave jump
            base_f = 110.0 if (beat_num < 2) else 130.8
            bass1 = synthesize_tone(base_f, beat_dur / 2.2, sr)
            bass2 = synthesize_tone(base_f * 2, beat_dur / 2.2, sr)
            overlay(track, bass1, int(cur_t * sr), gain=0.4)
            overlay(track, bass2, int((cur_t + beat_dur / 2.0) * sr), gain=0.35)
            cur_t += beat_dur

    elif genre == "hiphop":
        # Heavy sub-bass 808, 90 BPM, punchy snare on beats 2 & 4, fast hi-hats
        bpm = 90
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        while cur_t < duration:
            overlay(track, synthesize_kick(sr, duration=0.4), int(cur_t * sr), gain=0.8)
            beat_num = int(round(cur_t / beat_dur)) % 4
            if beat_num in (1, 3):
                overlay(track, synthesize_snare(sr), int(cur_t * sr), gain=0.7)
            # Fast trap-style hi-hat rolls
            for sub in range(4):
                overlay(track, synthesize_hihat(sr, 0.04), int((cur_t + sub * beat_dur / 4.0) * sr), gain=0.3)
            # 808 sub bass (40-60 Hz sine wave)
            sub_bass = np.sin(2 * np.pi * 55.0 * np.linspace(0, beat_dur, int(sr * beat_dur))) * 0.7
            overlay(track, sub_bass, int(cur_t * sr), gain=0.6)
            cur_t += beat_dur

    elif genre == "jazz":
        # Swing rhythm, walking bass, complex 7th chords (maj7, m7), syncopated ride cymbal
        bpm = 115
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        jazz_notes = [130.8, 146.8, 164.8, 174.6, 196.0, 220.0, 246.9]
        while cur_t < duration:
            # Walking bass line
            note = np.random.choice(jazz_notes)
            bass = synthesize_tone(note, beat_dur * 0.85, sr)
            overlay(track, bass, int(cur_t * sr), gain=0.4)
            # Swing cymbal (ding ... ding-a-ding)
            overlay(track, synthesize_hihat(sr, 0.06), int(cur_t * sr), gain=0.25)
            overlay(track, synthesize_hihat(sr, 0.04), int((cur_t + beat_dur * 0.66) * sr), gain=0.2)
            # Seventh chords periodically
            if int(cur_t / beat_dur) % 4 == 0:
                chord = [261.6, 329.6, 392.0, 493.8]  # Cmaj7
                for note_f in chord:
                    overlay(track, synthesize_tone(note_f, beat_dur * 2, sr), int(cur_t * sr), gain=0.15)
            cur_t += beat_dur

    elif genre == "reggae":
        # 78 BPM, heavy one-drop kick on beat 3, chop guitar chords on beats 2 & 4
        bpm = 78
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        while cur_t < duration:
            beat_num = int(round(cur_t / beat_dur)) % 4
            # One drop (kick on 3)
            if beat_num == 2:
                overlay(track, synthesize_kick(sr, 0.35), int(cur_t * sr), gain=0.8)
                overlay(track, synthesize_snare(sr), int(cur_t * sr), gain=0.5)
            # Guitar chop on off-beats (skank)
            if beat_num in (1, 3):
                chord = [196.0, 246.9, 293.6]  # G major chop
                for f in chord:
                    overlay(track, synthesize_tone(f, 0.09, sr), int(cur_t * sr), gain=0.3)
            # Deep dub bassline
            overlay(track, synthesize_tone(65.4, beat_dur * 0.7, sr), int(cur_t * sr), gain=0.5)
            cur_t += beat_dur

    elif genre == "rock":
        # Driving 120 BPM beat: Kick on 1 & 3, Snare on 2 & 4, guitar riff in A minor
        bpm = 120
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        while cur_t < duration:
            beat_num = int(round(cur_t / beat_dur)) % 4
            if beat_num in (0, 2):
                overlay(track, synthesize_kick(sr), int(cur_t * sr), gain=0.7)
            else:
                overlay(track, synthesize_snare(sr), int(cur_t * sr), gain=0.6)
            # 8th note hi-hat
            overlay(track, synthesize_hihat(sr, 0.04), int(cur_t * sr), gain=0.2)
            overlay(track, synthesize_hihat(sr, 0.04), int((cur_t + beat_dur / 2.0) * sr), gain=0.2)
            # Overdriven rock guitar chords
            riff_f = 220.0 if beat_num < 2 else 261.6
            guitar = synthesize_tone(riff_f, beat_dur * 0.8, sr) + synthesize_tone(riff_f * 1.5, beat_dur * 0.8, sr)
            guitar = np.tanh(guitar * 2.2) * 0.4
            overlay(track, guitar, int(cur_t * sr), gain=0.5)
            cur_t += beat_dur

    elif genre == "pop":
        # 118 BPM, bright synth arpeggios, bright electro clap/snare, 4-beat pulse
        bpm = 118
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        pop_melody = [523.2, 587.3, 659.2, 783.9, 880.0]  # C5, D5, E5, G5, A5
        while cur_t < duration:
            overlay(track, synthesize_kick(sr), int(cur_t * sr), gain=0.6)
            beat_num = int(round(cur_t / beat_dur)) % 4
            if beat_num in (1, 3):
                overlay(track, synthesize_snare(sr), int(cur_t * sr), gain=0.5)
            # Bright sparkling synth arpeggio
            note = pop_melody[beat_num % len(pop_melody)]
            synth = synthesize_tone(note, beat_dur * 0.4, sr) + synthesize_tone(note * 2, beat_dur * 0.4, sr) * 0.3
            overlay(track, synth, int(cur_t * sr), gain=0.3)
            cur_t += beat_dur

    elif genre == "country":
        # 100 BPM acoustic strumming (boom-chick pattern), major triad acoustic tones
        bpm = 100
        beat_dur = 60.0 / bpm
        cur_t = 0.0
        while cur_t < duration:
            beat_num = int(round(cur_t / beat_dur)) % 4
            # Boom (low bass note) on 1 & 3
            if beat_num in (0, 2):
                bass_f = 110.0 if beat_num == 0 else 164.8
                overlay(track, synthesize_tone(bass_f, beat_dur * 0.7, sr), int(cur_t * sr), gain=0.5)
            else:
                # Chick (strum chord) on 2 & 4
                for f in [220.0, 277.2, 329.6]:
                    overlay(track, synthesize_tone(f, beat_dur * 0.35, sr), int(cur_t * sr), gain=0.25)
            cur_t += beat_dur

    # Normalize audio to [-0.95, 0.95] to prevent clipping
    max_val = np.max(np.abs(track))
    if max_val > 0:
        track = (track / max_val) * 0.95

    return track.astype(np.float32)


def generate_all_samples(
    tracks_per_genre: int = 5,
    duration: float = 12.0,
    dest_dir: Path = RAW_DATA_DIR
) -> Dict[str, List[Path]]:
    """
    Generate audio files for all 10 genres and save to the dataset folder.
    Returns a mapping of genre name to list of created file paths.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, List[Path]] = {g: [] for g in GENRES}

    print(f"Generating {tracks_per_genre} synthetic tracks per genre across {len(GENRES)} genres...")
    for genre in GENRES:
        genre_folder = dest_dir / genre
        genre_folder.mkdir(parents=True, exist_ok=True)

        for i in range(tracks_per_genre):
            audio = generate_genre_audio(genre, duration=duration, sr=SAMPLE_RATE)
            filepath = genre_folder / f"{genre}.{i:05d}.wav"
            sf.write(str(filepath), audio, SAMPLE_RATE)
            created[genre].append(filepath)

        # Also save one representative sample in SAMPLES_DIR for quick testing / GUI demo
        sample_file = SAMPLES_DIR / f"{genre}_demo.wav"
        if not sample_file.exists():
            sf.write(str(sample_file), audio, SAMPLE_RATE)

    print(f"Successfully generated {tracks_per_genre * len(GENRES)} audio tracks in: {dest_dir}")
    return created


if __name__ == "__main__":
    generate_all_samples(tracks_per_genre=6, duration=15.0)
