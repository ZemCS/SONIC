import os
import sys
import time
import threading
import warnings
import subprocess
import torch
from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
from pathlib import Path
from flask_cors import CORS

# Import our new modules
from config import config, UI
from database import fused_inferences
from pipeline import MoodViT, MultimodalMoodClassifier, extract_metadata, get_image_transform
from auth import auth_bp
from routes import routes_bp
from spotify_utils import global_sp_oauth, resolve_spotify_uris
from extensions import jwt, limiter
import state
import logging

# Suppress Flask/Werkzeug noise
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")

# Security bypass for torch version in transformers
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

app = Flask(__name__)

CORS(app)

# --- Extension Initialization ---
app.config["JWT_SECRET_KEY"] = config.JWT_SECRET_KEY
jwt.init_app(app)
limiter.init_app(app)

# --- Register Blueprints ---
app.register_blueprint(auth_bp)
app.register_blueprint(routes_bp)

# --- Serve Uploaded Images ---
from flask import send_from_directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(config.UPLOADS_DIR, filename)

# Ensure uploads directory exists
os.makedirs(config.UPLOADS_DIR, exist_ok=True)

def load_models():
    print(UI.info("Loading Image Model..."))
    try:
        device = config.DEVICE
        state.image_model = MoodViT(num_classes=len(config.IMAGE_CLASSES), dropout_rate=0.4)
        # Explicitly move to device first
        state.image_model.to(device)
        
        # Load state dict
        state_dict = torch.load(config.IMAGE_MODEL_PATH, map_location=device)
        state.image_model.load_state_dict(state_dict)
        
        # Ensure it's in eval mode and definitely on device
        state.image_model.to(device).eval()
        print(UI.success(f"Image Model Loaded on {device}."))
    except Exception as e:
        print(UI.error(f"Failed to load Image Model: {e}"))

    print(UI.info("Loading Multimodal Song Pipeline..."))
    try:
        state.song_pipeline = MultimodalMoodClassifier()
        print(UI.success("Multimodal Pipeline Loaded."))
    except Exception as e:
        print(UI.error(f"Failed to load Multimodal Pipeline: {e}"))

def build_song_database():
    if state.song_pipeline is None:
        print("Skipping song scan (Pipeline not loaded).")
        return

    target_dir = Path(config.SONGS_DIR)
    if not target_dir.exists():
        print(f"Warning: Song directory {target_dir} does not exist.")
        return

    print(UI.info(f"Scanning Songs in: {target_dir}"))
    valid_extensions = [".mp3", ".wav", ".flac", ".m4a"]
    files = [f for f in target_dir.rglob("*") if f.suffix.lower() in valid_extensions]

    existing_filenames = {song["filename"] for song in state.SONG_DATABASE}
    new_songs_count = 0

    for file_path in files:
        filename = file_path.name
        if filename in existing_filenames:
            continue

        try:
            cached_inference = fused_inferences.find_one({"filename": filename})
            if cached_inference:
                title, artist = extract_metadata(str(file_path))
                mood = cached_inference["predicted_mood"].lower()
                conf = cached_inference["confidence"]

                # NEW: Update DB if artist was previously 'Unknown Artist' but is now resolved
                if cached_inference.get("artist") == "Unknown Artist" and artist != "Unknown Artist":
                    fused_inferences.update_one(
                        {"_id": cached_inference["_id"]},
                        {"$set": {"title": title, "artist": artist}}
                    )
                    print(UI.info(f"  [Refinement] Updated metadata for {filename}: {artist} - {title}"))

                state.SONG_DATABASE.append({
                    "title": title,
                    "artist": artist,
                    "mood": mood,
                    "filename": filename,
                    "confidence": conf,
                    "spotify_uri": cached_inference.get("spotify_uri"),
                })
                existing_filenames.add(filename)
                continue

            print(UI.info(f"  [Analyzing] {filename} (Not in cache)..."))
            result = state.song_pipeline.analyze_track(str(file_path))
            
            # Save to MongoDB
            lyrics_data = result.get("Individual", {}).get("lyrics", {})
            audio_data = result.get("Individual", {}).get("audio", {})
            fused_data = result.get("Results", {})
            va_data = result.get("VA", {})

            from database import audio_inferences, lyrics_inferences
            
            lyrics_inferences.insert_one({
                "filename": filename,
                "user_id": "system",
                "lyrics": lyrics_data.get("text"),
                "original_lyrics": lyrics_data.get("original_text"),
                "translated_lyrics": lyrics_data.get("translated_text"),
                "language": lyrics_data.get("language"),
                "mood": lyrics_data.get("mood"),
                "confidence": lyrics_data.get("confidence"),
                "source": lyrics_data.get("source"),
                "timestamp": datetime.now()
            })

            audio_inferences.insert_one({
                "filename": filename,
                "user_id": "system",
                "mood": audio_data.get("mood"),
                "confidence": audio_data.get("confidence"),
                "valence": va_data.get("valence"),
                "arousal": va_data.get("arousal"),
                "timestamp": datetime.now()
            })

            title, artist = extract_metadata(str(file_path))
            fused_inferences.insert_one({
                "filename": filename,
                "user_id": "system",
                "title": title,
                "artist": artist,
                "predicted_mood": fused_data.get("Predicted Mood"),
                "confidence": fused_data.get("Confidence"),
                "timestamp": datetime.now()
            })

            title, artist = extract_metadata(str(file_path))
            mood = result["Results"]["Predicted Mood"].lower()
            conf = result["Results"]["Confidence"]
            state.SONG_DATABASE.append({
                "title": title, "artist": artist, "mood": mood,
                "filename": filename, "confidence": conf, "spotify_uri": None,
            })
            new_songs_count += 1
            existing_filenames.add(filename)

        except Exception as e:
            print(UI.error(f"Failed to process {filename}: {e}"))

    print(UI.success(f"Database sync complete. {len(state.SONG_DATABASE)} total, {new_songs_count} new."))
    resolve_spotify_uris()

@app.route("/")
def index():
    return jsonify({"status": "running", "service": "SONIC Backend"})

def _free_port(port):
    """Kill any existing process on the given port before starting."""
    import subprocess, signal
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGKILL)
        if any(pids):
            print(UI.warning(f"Killed existing process(es) on port {port}."))
    except Exception:
        pass


def launch_ngrok_process():
    # Clear the old ngrok URL file to ensure we wait for a fresh one
    ngrok_file = Path(__file__).resolve().parent / "ngrok_url.txt"
    if ngrok_file.exists():
        ngrok_file.unlink()
    
    script_path = Path(__file__).resolve().parent / "start_ngrok.py"
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(UI.info("Launching ngrok subprocess..."))
    return proc


def wait_for_ngrok_url(timeout=30):
    ngrok_file = Path(__file__).resolve().parent / "ngrok_url.txt"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ngrok_file.exists():
            url = ngrok_file.read_text().strip()
            if url and not url.startswith("ERROR"):
                return url
        time.sleep(1)
    raise TimeoutError("Timed out waiting for ngrok_url.txt to contain a valid URL.")


def update_spotify_redirect_uri():
    """Update Spotify redirect URI in .env file with current ngrok URL."""
    ngrok_file = Path(__file__).resolve().parent / "ngrok_url.txt"
    env_file = Path(__file__).resolve().parent / ".env"
    
    if not ngrok_file.exists():
        print(UI.warning("ngrok_url.txt not found. Skipping Spotify redirect URI update."))
        return
    
    try:
        with open(ngrok_file, "r") as f:
            ngrok_url = f.read().strip()
        
        if not ngrok_url or "ERROR" in ngrok_url:
            print(UI.warning(f"Invalid ngrok URL: {ngrok_url}. Skipping update."))
            return
        
        # Remove any trailing slash and ensure it's just the base URL
        base_url = ngrok_url.rstrip('/')
        new_redirect_uri = f"{base_url}/spotify/callback"
        
        # Read current .env content
        with open(env_file, "r") as f:
            lines = f.readlines()
        
        # Update or add the SPOTIFY_REDIRECT_URI line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("SPOTIFY_REDIRECT_URI="):
                lines[i] = f"SPOTIFY_REDIRECT_URI={new_redirect_uri}\n"
                updated = True
                break
        
        if not updated:
            lines.append(f"SPOTIFY_REDIRECT_URI={new_redirect_uri}\n")
        
        # Write back to .env
        with open(env_file, "w") as f:
            f.writelines(lines)
        
        print(UI.success(f"Updated Spotify redirect URI to: {new_redirect_uri}"))
        
        # Reload environment variables to pick up the changes
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        # Update the config object with the new value
        import config
        config.config.SPOTIFY_REDIRECT_URI = new_redirect_uri
        
        # Update the global Spotify OAuth object
        import spotify_utils
        spotify_utils.global_sp_oauth = spotify_utils.get_spotify_oauth()
            
    except Exception as e:
        print(UI.error(f"Failed to update Spotify redirect URI: {e}"))

if __name__ == "__main__":
    ngrok_process = launch_ngrok_process()
    try:
        wait_for_ngrok_url()
    except Exception as e:
        print(UI.error(f"Ngrok startup failed: {e}"))
        ngrok_process.terminate()
        sys.exit(1)

    update_spotify_redirect_uri()
    _free_port(5000)
    load_models()
    threading.Thread(target=build_song_database, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
