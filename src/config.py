import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LINK_FILE = ROOT_DIR / "Link.txt"
FONTS_DIR = ROOT_DIR / "fonts"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"
LOCAL_KEYS_FILE = ROOT_DIR / "openrouter_keys.txt"


def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def env_int(name: str, default: int) -> int:
    val = env_str(name)
    return int(val) if val else default


def env_float(name: str, default: float) -> float:
    val = env_str(name)
    return float(val) if val else default


def env_bool(name: str, default: bool = False) -> bool:
    val = env_str(name).lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def load_openrouter_keys() -> list[str]:
    """
    একাধিক OpenRouter API Key লোড করে (প্রতি লাইনে একটি করে এন্টার দেওয়া)।
    নিরাপত্তার জন্য প্রথমে GitHub Secret (OPENROUTER_API_KEYS) থেকে নেবে,
    না থাকলে লোকাল openrouter_keys.txt ফাইল চেক করবে।
    """
    raw = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if not raw and LOCAL_KEYS_FILE.exists():
        raw = LOCAL_KEYS_FILE.read_text(encoding="utf-8")

    keys: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        for part in cleaned.split(","):
            k = part.strip()
            if k and k not in keys:
                keys.append(k)

    if not keys:
        raise ValueError(
            "কোনো OpenRouter API Key পাওয়া যায়নি! "
            "GitHub Secret 'OPENROUTER_API_KEYS'-এ একটার নিচে একটা Key দিন।"
        )
    return keys


def load_supadata_keys() -> list[str]:
    """
    একাধিক Supadata API Key লোড করে (প্রতি লাইনে একটি করে এন্টার দেওয়া)।
    """
    raw = os.getenv("SUPADATA_API_KEYS", "").strip()
    keys: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        for part in cleaned.split(","):
            k = part.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


DEFAULT_OPENROUTER_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
    "google/gemini-4-neon",
    "deepseek/deepseek-v3.2",
    "google/gemini-2.5-pro",
    "openai/gpt-4o-mini",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
]


def load_openrouter_models() -> list[str]:
    custom = os.getenv("OPENROUTER_MODELS", "").strip()
    if custom:
        models = [
            m.strip()
            for line in custom.splitlines()
            for m in line.split(",")
            if m.strip() and not m.strip().startswith("#")
        ]
        if models:
            return models

    if env_bool("OPENROUTER_FREE_ONLY", False):
        return [m for m in DEFAULT_OPENROUTER_MODELS if m.endswith(":free") or m == "openrouter/free"]

    return list(DEFAULT_OPENROUTER_MODELS)


# --- Video & Background Gameplay Config ---
VIDEO_WIDTH = env_int("VIDEO_WIDTH", 1920)
VIDEO_HEIGHT = env_int("VIDEO_HEIGHT", 1080)
VIDEO_FPS = env_int("VIDEO_FPS", 30)
FFMPEG_PRESET = env_str("FFMPEG_PRESET", "faster")
BG_DELETE_THRESHOLD_SEC = env_float("BG_DELETE_THRESHOLD_SEC", 60.0)
ALLOW_BG_LOOP = env_bool("ALLOW_BG_LOOP", True)

# --- TTS Config (Permanent Kaggle Account Cache + Dual T4 GPU Qwen3-TTS) ---
TTS_ENGINE = env_str("TTS_ENGINE", "auto")
KAGGLE_USERNAME = env_str("KAGGLE_USERNAME", "")
KAGGLE_KEY = env_str("KAGGLE_KEY", "")
KAGGLE_API_TOKEN = env_str("KAGGLE_API_TOKEN", "")
KAGGLE_CACHE_KERNEL_SLUG = env_str("KAGGLE_CACHE_KERNEL_SLUG", "qwen3-tts-model-cache")
KAGGLE_KERNEL_SLUG = env_str("KAGGLE_KERNEL_SLUG", "yt-monologue-qwen3-tts-worker")
KAGGLE_TIMEOUT_SEC = env_int("KAGGLE_TIMEOUT_SEC", 1500)

QWEN_TTS_MODEL = env_str("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
QWEN_SPEAKER = env_str("QWEN_SPEAKER", "Ryan")
QWEN_INSTRUCT = env_str(
    "QWEN_INSTRUCT",
    "Deep, calm, emotionally gripping philosophical male narrator speaking with warm authority and natural pauses.",
)

EDGE_TTS_VOICE = env_str("EDGE_TTS_VOICE", "en-US-AndrewMultilingualNeural")
EDGE_TTS_RATE = env_str("EDGE_TTS_RATE", "-4%")
KOKORO_VOICE = env_str("KOKORO_VOICE", "am_michael")
KOKORO_SPEED = env_float("KOKORO_SPEED", 0.96)

# --- Transcript & Network Config ---
PROXY_URL = env_str("PROXY_URL", "")
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "").strip()

# --- YouTube Upload Config ---
YOUTUBE_CLIENT_ID = env_str("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET = env_str("YOUTUBE_CLIENT_SECRET", "")
YOUTUBE_REFRESH_TOKEN = env_str("YOUTUBE_REFRESH_TOKEN", "")
YOUTUBE_PRIVACY_STATUS = env_str("YOUTUBE_PRIVACY_STATUS", "public")
YOUTUBE_CATEGORY_ID = env_str("YOUTUBE_CATEGORY_ID", "22")
SKIP_UPLOAD = env_bool("SKIP_UPLOAD", False)

# --- Upload Mode Switch & Google Drive (rclone) Config ---
UPLOAD_TO_YOUTUBE = env_bool("UPLOAD_TO_YOUTUBE", False)
CLEANUP_AFTER_GDRIVE = env_bool("CLEANUP_AFTER_GDRIVE", False)
RCLONE_CONFIG = os.getenv("RCLONE_CONFIG", "").strip()
RCLONE_REMOTE_NAME = env_str("RCLONE_REMOTE_NAME", "")
GDRIVE_FOLDER_ID = env_str("GDRIVE_FOLDER_ID", "")
GDRIVE_BG_FOLDER_ID = env_str("GDRIVE_BG_FOLDER_ID", "")
