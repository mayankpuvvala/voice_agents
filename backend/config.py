"""Runtime configuration, loaded from .env / environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _optional_float(name: str, default: str) -> float | None:
    """Returns None when explicitly blanked — reasoning models reject `temperature`."""
    raw = os.environ.get(name, default).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class Settings:
    # Paths
    root: Path = ROOT
    db_path: Path = ROOT / "data" / "receptionist.db"
    knowledge_dir: Path = ROOT / "knowledge"
    frontend_dir: Path = ROOT / "frontend"
    piper_data_dir: Path = ROOT / "data" / "piper-voices"

    # Chat model — OPENAI_BASE_URL can point this at any OpenAI-compatible
    # gateway (Azure, Groq, etc.) without a code change.
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float | None = _optional_float("OPENAI_TEMPERATURE", "0.4")

    # Speech to text. Groq's hosted Whisper is the default: free-tier,
    # telephony-grade accuracy, and it offloads transcription off this
    # process's CPU entirely. `faster-whisper` stays as an offline fallback.
    stt_provider: str = os.environ.get("STT_PROVIDER", "groq")  # "groq" | "local"
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_base_url: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_stt_model: str = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    # Multilingual model (not the `.en` variants) — this app answers in
    # English and Hindi, so the fallback path needs language coverage too.
    whisper_model: str = os.environ.get("WHISPER_MODEL", "small")
    whisper_compute_type: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    # Text to speech. Piper is self-hosted and free at any call volume, and
    # it's the only option of the two that can ever run on a real phone call
    # (the browser's speechSynthesis cannot). The frontend falls back to
    # speechSynthesis automatically when this is off.
    tts_enabled: bool = _bool("TTS_ENABLED", True)
    piper_voice_en: str = os.environ.get("PIPER_VOICE_EN", "en_US-lessac-medium")
    piper_voice_hi: str = os.environ.get("PIPER_VOICE_HI", "hi_IN-rohan-medium")

    # Business identity
    business_name: str = os.environ.get("BUSINESS_NAME", "Spice Route Kitchen")
    # Real IANA zone, used for every date/time computation (reservations,
    # "today" boundaries). timezone_label is display-only.
    business_timezone: str = os.environ.get("BUSINESS_TIMEZONE", "Asia/Kolkata")
    timezone_label: str = os.environ.get("BUSINESS_TIMEZONE_LABEL", "IST")
    receptionist_name: str = os.environ.get("RECEPTIONIST_NAME", "Meera")
    greeting: str = os.environ.get(
        "GREETING",
        "Thanks for calling Spice Route Kitchen, this is Meera — how can I help you "
        "today? Aap Hindi mein bhi baat kar sakte hain.",
    )

    # Admin dashboard / API — required once this leaves 127.0.0.1. Blank
    # means "no auth", which is only acceptable for local development.
    admin_username: str = os.environ.get("ADMIN_USERNAME", "")
    admin_password: str = os.environ.get("ADMIN_PASSWORD", "")

    # Server
    host: str = os.environ.get("HOST", "127.0.0.1")
    port: int = _int("PORT", 8000)

    @property
    def auth_required(self) -> bool:
        return bool(self.admin_username and self.admin_password)


settings = Settings()

settings.db_path.parent.mkdir(parents=True, exist_ok=True)
settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
settings.piper_data_dir.mkdir(parents=True, exist_ok=True)
