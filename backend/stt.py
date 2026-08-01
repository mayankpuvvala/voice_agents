"""Speech-to-text: Groq-hosted Whisper (primary) with local faster-whisper fallback.

Groq's Whisper endpoint is OpenAI-compatible and multilingual (English and
Hindi are both well covered), and it offloads transcription off this
process's CPU — its free tier is generous enough for the small number of
concurrent calls this app targets. `faster-whisper` stays as an offline
fallback so a Groq outage or a missing API key degrades transcription
instead of taking calls down.

The browser sends one complete WebM/Opus blob per utterance, which both
paths can decode directly.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading

from openai import AsyncOpenAI

from .config import settings

log = logging.getLogger("receptionist.stt")

_local_model = None
_local_lock = threading.Lock()

_groq_client: AsyncOpenAI | None = None


def _load_local():
    global _local_model
    with _local_lock:
        if _local_model is None:
            from faster_whisper import WhisperModel

            _local_model = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type=settings.whisper_compute_type,
            )
        return _local_model


def _get_groq_client() -> AsyncOpenAI | None:
    global _groq_client
    if not settings.groq_api_key:
        return None
    if _groq_client is None:
        _groq_client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _groq_client


def warm_up() -> None:
    """Load the local fallback model ahead of the first caller.

    Loaded regardless of which provider is primary, so a Groq outage
    degrades to an already-warm local model rather than a cold one.
    """
    _load_local()


def _transcribe_local_sync(audio: bytes) -> str:
    model = _load_local()
    segments, _info = model.transcribe(
        io.BytesIO(audio),
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _transcribe_groq(audio: bytes) -> str:
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("No GROQ_API_KEY configured")
    buf = io.BytesIO(audio)
    buf.name = "utterance.webm"
    response = await client.audio.transcriptions.create(
        model=settings.groq_stt_model,
        file=buf,
    )
    return (response.text or "").strip()


async def transcribe(audio: bytes) -> str:
    """Transcribe one utterance. Returns "" when nothing intelligible was said."""
    if len(audio) < 2000:  # a fraction of a second of silence
        return ""

    if settings.stt_provider == "groq" and settings.groq_api_key:
        try:
            return await _transcribe_groq(audio)
        except Exception:
            log.exception("Groq transcription failed — falling back to local Whisper")

    return await asyncio.to_thread(_transcribe_local_sync, audio)
