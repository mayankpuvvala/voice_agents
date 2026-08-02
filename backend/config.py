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
    recordings_dir: Path = ROOT / "data" / "recordings"

    # Chat model — OPENAI_BASE_URL can point this at any OpenAI-compatible
    # gateway (Azure, Groq, etc.) without a code change. Resolved here rather
    # than left for the openai SDK's own fallback: the SDK does
    # `base_url = base_url or os.environ.get("OPENAI_BASE_URL")`, and
    # os.environ.get returns "" (not None) for a blank-but-set variable —
    # which the SDK then treats as a real override instead of falling
    # through to its default, breaking the connection entirely.
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float | None = _optional_float("OPENAI_TEMPERATURE", "0.4")
    base_url: str = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"

    # Speech to text. "openai" (gpt-4o-mini-transcribe) is the most accurate
    # and least prone to Whisper's silence/noise hallucinations, falling
    # back to Groq's hosted Whisper (cheaper, still solid) and then local
    # faster-whisper if both cloud providers are unreachable. Needs its own
    # real platform.openai.com key — OPENAI_API_KEY may be pointed at Groq
    # for the chat model, so it can't be reused here.
    stt_provider: str = os.environ.get("STT_PROVIDER", "openai")  # "openai" | "groq" | "local"
    openai_stt_api_key: str = os.environ.get("OPENAI_STT_API_KEY", "")
    openai_stt_model: str = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    # English-only for now — language auto-detection was mis-firing on short
    # or ambiguous audio (a spoken name, background noise) and transcribing
    # it as Chinese/Arabic/etc instead of returning nothing. Blank reverts
    # to each provider's own auto-detection.
    stt_language: str = os.environ.get("STT_LANGUAGE", "en")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_base_url: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_stt_model: str = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    # Multilingual model (not the `.en` variants) — this app answers in
    # English and Hindi, so the fallback path needs language coverage too.
    whisper_model: str = os.environ.get("WHISPER_MODEL", "small")
    whisper_compute_type: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    # Text to speech. Piper (self-hosted, $0/mo) is the default and the only
    # option that can ever run on a real phone call (the browser's
    # speechSynthesis cannot). "openai" swaps in gpt-4o-mini-tts for more
    # natural speech at a small per-call cost — it reuses the STT key above
    # unless OPENAI_TTS_API_KEY is set separately. The frontend falls back to
    # speechSynthesis automatically whenever TTS_ENABLED is off or synthesis
    # fails for a given sentence.
    tts_enabled: bool = _bool("TTS_ENABLED", True)
    tts_provider: str = os.environ.get("TTS_PROVIDER", "piper")  # "piper" | "openai"
    piper_voice_en: str = os.environ.get("PIPER_VOICE_EN", "en_US-lessac-medium")
    piper_voice_hi: str = os.environ.get("PIPER_VOICE_HI", "hi_IN-rohan-medium")
    openai_tts_api_key: str = os.environ.get("OPENAI_TTS_API_KEY", "") or os.environ.get(
        "OPENAI_STT_API_KEY", ""
    )
    openai_tts_model: str = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    openai_tts_voice: str = os.environ.get("OPENAI_TTS_VOICE", "marin")

    # Call recording — both sides get merged into one WAV per call, saved
    # under data/recordings/ and playable from the admin dashboard. Off
    # switch is here for jurisdictions where recording calls needs caller
    # disclosure/consent; if left on, make sure the greeting says calls may
    # be recorded.
    call_recording_enabled: bool = _bool("CALL_RECORDING_ENABLED", True)

    # Business identity
    business_name: str = os.environ.get("BUSINESS_NAME", "Spice Route Kitchen")
    # Real IANA zone, used for every date/time computation (reservations,
    # "today" boundaries). timezone_label is display-only.
    business_timezone: str = os.environ.get("BUSINESS_TIMEZONE", "Asia/Kolkata")
    timezone_label: str = os.environ.get("BUSINESS_TIMEZONE_LABEL", "IST")
    receptionist_name: str = os.environ.get("RECEPTIONIST_NAME", "Meera")
    greeting: str = os.environ.get(
        "GREETING",
        "Thanks for calling Spice Kitchen, this is Meera — how can I help you today?",
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
settings.recordings_dir.mkdir(parents=True, exist_ok=True)
