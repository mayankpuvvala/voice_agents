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
    root: Path = ROOT
    db_path: Path = ROOT / "data" / "receptionist.db"
    knowledge_dir: Path = ROOT / "knowledge"
    frontend_dir: Path = ROOT / "frontend"
    piper_data_dir: Path = ROOT / "data" / "piper-voices"
    recordings_dir: Path = ROOT / "data" / "recordings"

    # Resolved here, not left to the openai SDK's own env fallback: a blank
    # OPENAI_BASE_URL reads back as "" (not None) and the SDK treats that as
    # a real override, breaking the connection.
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float | None = _optional_float("OPENAI_TEMPERATURE", "0.4")
    base_url: str = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"

    # STT: openai (gpt-4o-mini-transcribe) -> groq -> local faster-whisper.
    # Needs its own platform.openai.com key since OPENAI_API_KEY may be
    # pointed at Groq for the chat model.
    stt_provider: str = os.environ.get("STT_PROVIDER", "openai")  # "openai" | "groq" | "local"
    openai_stt_api_key: str = os.environ.get("OPENAI_STT_API_KEY", "")
    openai_stt_model: str = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    # English-only for now — auto-detection misfired on short/ambiguous
    # audio, transcribing it as Chinese/Arabic/etc instead of nothing.
    stt_language: str = os.environ.get("STT_LANGUAGE", "en")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_base_url: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_stt_model: str = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    whisper_model: str = os.environ.get("WHISPER_MODEL", "small")
    whisper_compute_type: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    # TTS: Piper is self-hosted/free and the only option a real phone call
    # can use (browser speechSynthesis can't). "openai" swaps in
    # gpt-4o-mini-tts for a small per-call cost.
    tts_enabled: bool = _bool("TTS_ENABLED", True)
    tts_provider: str = os.environ.get("TTS_PROVIDER", "piper")  # "piper" | "openai"
    piper_voice_en: str = os.environ.get("PIPER_VOICE_EN", "en_US-lessac-medium")
    piper_voice_hi: str = os.environ.get("PIPER_VOICE_HI", "hi_IN-rohan-medium")
    openai_tts_api_key: str = os.environ.get("OPENAI_TTS_API_KEY", "") or os.environ.get(
        "OPENAI_STT_API_KEY", ""
    )
    openai_tts_model: str = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    openai_tts_voice: str = os.environ.get("OPENAI_TTS_VOICE", "marin")

    # Merged into one WAV per call under data/recordings/. Off switch is for
    # jurisdictions requiring caller consent to record.
    call_recording_enabled: bool = _bool("CALL_RECORDING_ENABLED", True)

    # Notified on reservation create/update/cancel to sync a Google Calendar
    # event. Blank disables it; a failed/unconfigured webhook is non-fatal.
    n8n_reservation_webhook_url: str = os.environ.get("N8N_RESERVATION_WEBHOOK_URL", "")

    business_name: str = os.environ.get("BUSINESS_NAME", "Spice Route Kitchen")
    business_timezone: str = os.environ.get("BUSINESS_TIMEZONE", "Asia/Kolkata")
    timezone_label: str = os.environ.get("BUSINESS_TIMEZONE_LABEL", "IST")
    receptionist_name: str = os.environ.get("RECEPTIONIST_NAME", "Meera")
    greeting: str = os.environ.get(
        "GREETING",
        "Thanks for calling Spice Kitchen, this is Meera — how can I help you today?",
    )

    # Blank means no auth — only acceptable for local development.
    admin_username: str = os.environ.get("ADMIN_USERNAME", "")
    admin_password: str = os.environ.get("ADMIN_PASSWORD", "")

    host: str = os.environ.get("HOST", "127.0.0.1")
    port: int = _int("PORT", 8000)

    @property
    def auth_required(self) -> bool:
        return bool(self.admin_username and self.admin_password)


settings = Settings()

settings.db_path.parent.mkdir(parents=True, exist_ok=True)
settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
settings.piper_data_dir.mkdir(parents=True, exist_ok=True)
settings.recordings_dir.mkdir(parents=True, exist_ok=True)
