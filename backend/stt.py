"""Local speech-to-text with faster-whisper.

The browser sends one complete WebM/Opus blob per utterance, so faster-whisper
can decode it directly (it bundles PyAV) — no ffmpeg install required.
"""

from __future__ import annotations

import asyncio
import io
import threading

from .config import settings

_model = None
_model_lock = threading.Lock()


def _load():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type=settings.whisper_compute_type,
            )
        return _model


def warm_up() -> None:
    """Download/load the model ahead of the first caller."""
    _load()


def _transcribe_sync(audio: bytes) -> str:
    model = _load()
    segments, _info = model.transcribe(
        io.BytesIO(audio),
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def transcribe(audio: bytes) -> str:
    """Transcribe one utterance. Returns "" when nothing intelligible was said."""
    if len(audio) < 2000:  # a fraction of a second of silence
        return ""
    return await asyncio.to_thread(_transcribe_sync, audio)
