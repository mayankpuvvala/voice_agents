"""Text-to-speech: self-hosted Piper (default, free at any call volume) or
OpenAI's gpt-4o-mini-tts for more natural speech at a small per-call cost.

Voice is picked per reply by a cheap Unicode-script check on the *generated*
text (Devanagari => Hindi voice), rather than trusting whatever language the
caller spoke — that way a mismatched voice/text can't happen even if the
model doesn't follow the "reply in the caller's language" instruction. Piper
needs this since its voices are monolingual; OpenAI's voices handle both
languages themselves; so the check only feeds the Piper path.

Piper voice models are downloaded on first use (same pattern as
`stt.warm_up` downloading the Whisper model): if the .onnx file isn't
already in `settings.piper_data_dir`, `python -m piper.download_voices`
fetches it.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import wave
from io import BytesIO

from .config import settings

log = logging.getLogger("receptionist.tts")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

_voices: dict[str, object] = {}
_lock = threading.Lock()

_openai_client = None
_openai_lock = threading.Lock()

# See the matching comment in stt.py: the openai SDK reads OPENAI_BASE_URL
# from the environment whenever base_url isn't passed explicitly, and an
# empty-but-set value leaks through as a real override. Pinning this here
# keeps TTS on real OpenAI regardless of what the chat client's base_url is.
_OPENAI_BASE_URL = "https://api.openai.com/v1"


def detect_voice_language(text: str) -> str:
    """"en" or "hi", based on the script the reply is actually written in."""
    return "hi" if _DEVANAGARI.search(text) else "en"


def _voice_id(lang: str) -> str:
    return settings.piper_voice_hi if lang == "hi" else settings.piper_voice_en


def _ensure_downloaded(voice_id: str) -> None:
    onnx_path = settings.piper_data_dir / f"{voice_id}.onnx"
    if onnx_path.exists():
        return
    log.info("Downloading Piper voice '%s' (first use)...", voice_id)
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", voice_id],
        cwd=str(settings.piper_data_dir),
        check=True,
        capture_output=True,
    )


def _load(lang: str):
    with _lock:
        voice = _voices.get(lang)
        if voice is not None:
            return voice

    voice_id = _voice_id(lang)
    _ensure_downloaded(voice_id)

    from piper import PiperVoice

    onnx_path = settings.piper_data_dir / f"{voice_id}.onnx"
    voice = PiperVoice.load(str(onnx_path))
    with _lock:
        _voices[lang] = voice
    return voice


def _get_openai_client():
    global _openai_client
    if not settings.openai_tts_api_key:
        return None
    with _openai_lock:
        if _openai_client is None:
            from openai import OpenAI

            _openai_client = OpenAI(
                api_key=settings.openai_tts_api_key, base_url=_OPENAI_BASE_URL
            )
        return _openai_client


def warm_up() -> None:
    """Load the Piper voice ahead of the first caller.

    English only for now, matching the STT/agent language scope. No-ops for
    the OpenAI provider — there's no local model to preload.
    """
    if not settings.tts_enabled or settings.tts_provider != "piper":
        return
    try:
        _load("en")
    except Exception:
        log.exception("Could not load the Piper voice — TTS will fail over to the browser")


def _synthesize_piper(text: str) -> bytes:
    lang = detect_voice_language(text)
    voice = _load(lang)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buf.getvalue()


def _synthesize_openai(text: str) -> bytes:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("No OPENAI_TTS_API_KEY (or OPENAI_STT_API_KEY) configured")
    response = client.audio.speech.create(
        model=settings.openai_tts_model,
        voice=settings.openai_tts_voice,
        input=text,
        response_format="wav",
    )
    return response.content


def synthesize(text: str) -> bytes:
    """Render `text` to WAV bytes with the configured engine.

    Raises on failure — callers should treat that as "no server audio for
    this sentence" and let the frontend fall back to speechSynthesis rather
    than dropping the turn.
    """
    if settings.tts_provider == "openai":
        return _synthesize_openai(text)
    return _synthesize_piper(text)
