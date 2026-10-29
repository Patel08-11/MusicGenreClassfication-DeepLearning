"""
SONIX AI Studio Web Application Server.
Ultra-resilient server using Python's ThreadingHTTPServer.
Runs with zero external web framework dependencies, ensuring 100% reliable startup.
Features:
- Live PyTorch Deep Learning Genre Classification
- Real-Time Audio Streaming for Web Audio API
- Interactive Spectrogram Dashboard Generator
- Metrics & Output Files Delivery
"""

import sys
import os
import json
import cgi
import io
import mimetypes
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from features.extractor import load_audio, extract_audio_summary
from features.visualizer import plot_audio_dashboard
from predict import predict_genre

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
TEMP_UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class SonixStudioHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. Main Page
        if path == '/' or path == '/index.html':
            index_file = TEMPLATES_DIR / "index.html"
            if index_file.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(index_file.read_bytes())
                return

        # 2. Static Assets (/static/...)
        if path.startswith('/static/'):
            relative_path = path[len('/static/'):].lstrip('/')
            file_path = STATIC_DIR / relative_path
            if file_path.exists() and file_path.is_file():
                self.serve_file(file_path)
                return

        # 3. API: Preset Tracks (/api/tracks)
        if path == '/api/tracks':
            tracks = []
            genre_meta = {
                "blues": {"bpm": 105, "mood": "Soulful & Melancholy", "desc": "12-bar blues shuffle, minor pentatonic riffs"},
                "classical": {"bpm": 72, "mood": "Harmonic & Expressive", "desc": "Gentle strings arpeggios, sustained acoustic chords"},
                "country": {"bpm": 100, "mood": "Acoustic & Rustic", "desc": "Boom-chick rhythm, major triad acoustic guitar strum"},
                "disco": {"bpm": 124, "mood": "Energetic & Groovy", "desc": "Four-on-the-floor kick, octave bass, upbeat open hats"},
                "hiphop": {"bpm": 90, "mood": "Heavy Sub & Rhythmic", "desc": "Punchy 808 sub-bass, boom-bap snares, rapid trap rolls"},
                "jazz": {"bpm": 115, "mood": "Syncopated & Sophisticated", "desc": "Swing ride cymbal, walking bass, rich 7th chords"},
                "metal": {"bpm": 150, "mood": "Aggressive & Distorted", "desc": "Double-kick blast, overdriven power chords, high energy"},
                "pop": {"bpm": 118, "mood": "Catchy & Polished", "desc": "Bright synth hooks, 4-beat electro claps, melodic drive"},
                "reggae": {"bpm": 78, "mood": "Laid-Back & Syncopated", "desc": "One-drop kick on beat 3, guitar skank on off-beats"},
                "rock": {"bpm": 120, "mood": "Driving & Powerful", "desc": "Rock drums on 2 & 4, overdriven guitar riff, solid bass"}
            }

            for genre in GENRES:
                meta = genre_meta.get(genre, {"bpm": 120, "mood": "Musical", "desc": f"{genre.title()} track"})
                tracks.append({
                    "id": genre,
                    "genre": genre,
                    "title": f"{genre.title()} Horizon",
                    "artist": f"Studio Synth ({genre.upper()})",
                    "bpm": meta["bpm"],
                    "mood": meta["mood"],
                    "description": meta["desc"],
                    "audio_url": f"/api/audio/{genre}",
                    "spectrogram_url": f"/api/spectrogram/{genre}"
                })

            self.send_json({"tracks": tracks, "total": len(tracks), "genres": GENRES})
            return

        # 4. API: Audio Stream (/api/audio/{genre})
        if path.startswith('/api/audio/'):
            track_id = path[len('/api/audio/'):].split('?')[0]
            demo_file = SAMPLES_DIR / f"{track_id}_demo.wav"
            if not demo_file.exists():
                candidates = list((RAW_DATA_DIR / track_id).glob("*.wav"))
                if candidates:
                    demo_file = candidates[0]
                else:
                    demo_file = TEMP_UPLOADS_DIR / track_id

            if demo_file.exists():
                self.serve_file(demo_file, mime_type="audio/wav")
                return

        # 5. API: Spectrogram Image (/api/spectrogram/{genre})
        if path.startswith('/api/spectrogram/'):
            track_id = path[len('/api/spectrogram/'):].split('?')[0]
            img_candidate = OUTPUTS_DIR / f"predict_{track_id}_demo.png"
            if not img_candidate.exists():
                img_candidate = OUTPUTS_DIR / f"predict_{track_id}.png"

            if not img_candidate.exists():
                audio_file = SAMPLES_DIR / f"{track_id}_demo.wav"
                if audio_file.exists():
                    y, sr = load_audio(audio_file)
                    img_candidate = plot_audio_dashboard(
                        y, sr,
                        title=f"{track_id.upper()} Spectral Analysis",
                        save_path=OUTPUTS_DIR / f"predict_{track_id}.png"
                    )

            if img_candidate and Path(img_candidate).exists():
                self.serve_file(Path(img_candidate), mime_type="image/png")
                return

        # 6. API: Output Files (/api/outputs/{filename})
        if path.startswith('/api/outputs/'):
            filename = path[len('/api/outputs/'):].split('?')[0]
            target_file = OUTPUTS_DIR / filename
            if target_file.exists():
                self.serve_file(target_file)
                return

        # 7. API: Neural Performance Metrics (/api/metrics)
        if path == '/api/metrics':
            report_file = OUTPUTS_DIR / "evaluation_report.txt"
            report_content = report_file.read_text(encoding='utf-8') if report_file.exists() else "No report available."

            history_file = OUTPUTS_DIR / "training_history.json"
            history_data = json.loads(history_file.read_text(encoding='utf-8')) if history_file.exists() else {}

            output_pngs = [p.name for p in OUTPUTS_DIR.glob("*.png")]

            self.send_json({
                "report": report_content,
                "history": history_data,
                "all_pngs": output_pngs
            })
            return

        # Fallback 404
        self.send_error(404, f"Resource not found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/predict':
            content_type = self.headers.get('Content-Type', '')
            query_path = None

            if 'multipart/form-data' in content_type:
                # Parse multipart
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
                )

                if 'audio_file' in form and form['audio_file'].filename:
                    file_item = form['audio_file']
                    filename = Path(file_item.filename).name
                    query_path = TEMP_UPLOADS_DIR / filename
                    with open(query_path, 'wb') as f:
                        f.write(file_item.file.read())
                elif 'track_id' in form:
                    track_id = form.getvalue('track_id')
                    query_path = SAMPLES_DIR / f"{track_id}_demo.wav"
            else:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode('utf-8')
                try:
                    data = json.loads(body)
                    track_id = data.get('track_id')
                    if track_id:
                        query_path = SAMPLES_DIR / f"{track_id}_demo.wav"
                except Exception:
                    pass

            if query_path and query_path.exists():
                try:
                    result = predict_genre(query_path, top_k=10, save_spectrogram=True)
                    img_filename = Path(result["spectrogram_image"]).name if result.get("spectrogram_image") else ""
                    self.send_json({
                        "status": "success",
                        "file_name": query_path.name,
                        "primary_genre": result["primary_genre"],
                        "confidence": result["confidence"],
                        "top_predictions": result["top_predictions"],
                        "all_probabilities": result["all_probabilities"],
                        "audio_summary": result["audio_summary"],
                        "spectrogram_image_url": f"/api/outputs/{img_filename}" if img_filename else None,
                        "device": str(DEVICE)
                    })
                    return
                except Exception as e:
                    self.send_json({"status": "error", "message": str(e)}, status=500)
                    return

            self.send_json({"status": "error", "message": "Audio file or track_id not found"}, status=400)
            return

        self.send_error(404, "Unknown endpoint")

    def serve_file(self, file_path: Path, mime_type: str = None):
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type is None:
                mime_type = "application/octet-stream"

        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Clean logging
        pass


def run_studio_server(port: int = 5000):
    server_address = ('127.0.0.1', port)
    httpd = ThreadingHTTPServer(server_address, SonixStudioHandler)
    print("=" * 65)
    print("SONIX AI STUDIO WORKSTATION SERVER")
    print(f"URL: http://127.0.0.1:{port}")
    print(f"Hardware Acceleration: {DEVICE}")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    run_studio_server(port=5000)
