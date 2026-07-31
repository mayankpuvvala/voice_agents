"""Runtime configuration, loaded from .env / environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _optional_float(name: str, default: str) -> float | None:
    """Returns None when explicitly blanked — reasoning models reject `temperature`."""
    raw = os.environ.get(name, default).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _days(name: str, default: str) -> set[int]:
    raw = os.environ.get(name, "").strip() or default
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out or {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class Settings:
    # Paths
    root: Path = ROOT
    db_path: Path = ROOT / "data" / "receptionist.db"
    knowledge_dir: Path = ROOT / "knowledge"
    frontend_dir: Path = ROOT / "frontend"

    # Model
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
    temperature: float | None = _optional_float("OPENAI_TEMPERATURE", "0.4")

    # Speech to text
    whisper_model: str = os.environ.get("WHISPER_MODEL", "base.en")
    whisper_compute_type: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    # Business identity
    business_name: str = os.environ.get("BUSINESS_NAME", "Northgate Dental")
    timezone_label: str = os.environ.get("BUSINESS_TIMEZONE_LABEL", "local time")
    receptionist_name: str = os.environ.get("RECEPTIONIST_NAME", "Ava")
    greeting: str = os.environ.get(
        "GREETING",
        "Thanks for calling. How can I help you today?",
    )

    # Booking rules
    open_hour: int = _int("OPEN_HOUR", 9)
    close_hour: int = _int("CLOSE_HOUR", 17)
    open_days: set[int] = field(default_factory=lambda: _days("OPEN_DAYS", "0,1,2,3,4"))
    slot_minutes: int = _int("SLOT_MINUTES", 30)

    # Server
    host: str = os.environ.get("HOST", "127.0.0.1")
    port: int = _int("PORT", 8000)

    @property
    def open_days_label(self) -> str:
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return ", ".join(names[d] for d in sorted(self.open_days))

    @property
    def hours_label(self) -> str:
        return f"{self.open_hour:02d}:00-{self.close_hour:02d}:00 {self.timezone_label}"


settings = Settings()

settings.db_path.parent.mkdir(parents=True, exist_ok=True)
settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
