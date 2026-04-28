import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress noisy library logs BEFORE loading them
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow: Suppress info and warning
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"


# --- ANSI Terminal Colors ---
class UI:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def success(msg):
        return f"{UI.GREEN}{UI.BOLD}✓ {msg}{UI.RESET}"

    @staticmethod
    def info(msg):
        return f"{UI.CYAN}ℹ {msg}{UI.RESET}"

    @staticmethod
    def error(msg):
        return f"{UI.RED}{UI.BOLD}✗ {msg}{UI.RESET}"

    @staticmethod
    def warning(msg):
        return f"{UI.YELLOW}⚠ {msg}{UI.RESET}"


# Load environment variables
load_dotenv()

# Set HF_TOKEN if available in .env
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path(__file__).resolve().parent

    # --- API Keys ---
    GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # --- Image Model Settings ---
    IMAGE_MODEL_PATH = str(
        BASE_DIR / "models" / "image_mood_model" / "image_model_v2.pth"
    )
    UPLOADS_DIR = str(BASE_DIR / "uploads")
    IMAGE_CLASSES = ["aggressive", "calm", "joyful", "sad", "energetic"]

    # --- Song Pipeline Paths ---
    SONGS_DIR = str(BASE_DIR / "music_library")
    TEXT_MODEL_PATH = str(BASE_DIR / "models" / "lyrics_mood_model")
    AUDIO_MODEL_PATH = str(BASE_DIR / "models" / "audio_mood_model")

    AUDIO_CLASSES = ["aggressive", "calm", "energetic", "joyful", "romantic", "sad"]
    LYRICS_CLASSES = ["aggressive", "calm", "energetic", "joyful", "romantic", "sad"]
    EMBEDDING_MODEL = str(BASE_DIR / "models" / "msd-musicnn-1.pb")
    VA_MODEL = str(BASE_DIR / "models" / "deam-msd-musicnn-2.pb")

    # --- Audio/Text Params ---
    AUDIO_SAMPLE_RATE = 24000
    MAX_TEXT_LENGTH = 512
    AUDIO_WINDOW_SEC = 15.0
    AUDIO_HOP_SEC = 5.0
    TEXT_WINDOW_TOKENS = 512
    TEXT_HOP_TOKENS = 256
    TEXT_BATCH_SIZE = 8
    AUDIO_BATCH_SIZE = 4

    MANUAL_WEIGHTS = {
        "aggressive": {"text": 0.30, "audio": 0.70},
        "calm": {"text": 0.30, "audio": 0.70},
        "energetic": {"text": 0.40, "audio": 0.60},
        "joyful": {"text": 0.7, "audio": 0.30},
        "romantic": {"text": 0.70, "audio": 0.30},
        "sad": {"text": 0.70, "audio": 0.30},
        "default": {"text": 0.55, "audio": 0.45},
    }

    # --- Spotify API ---
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_SECRET")
    SPOTIFY_REDIRECT_URI = os.getenv(
        "SPOTIFY_REDIRECT_URI",
        "https://desultory-damion-semineutral.ngrok-free.dev/spotify/callback",
    )
    SPOTIFY_SCOPE = (
        "user-read-recently-played user-modify-playback-state user-read-playback-state"
    )

    # --- SMTP Settings ---
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

    # --- JWT Settings ---
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-sonic-key")


config = Config()
