"""Text-to-speech: self-hosted Piper, free at any call volume.

Voice is picked per reply by a cheap Unicode-script check on the *generated*
text (Devanagari => Hindi voice), rather than trusting whatever language the
caller spoke — that way a mismatched voice/text can't happen even if the
model doesn't follow the "reply in the caller's language" instruction.

Voice models are downloaded on first use (same pattern as `stt.warm_up`
downloading the Whisper model): if the .onnx file isn't already in
`settings.piper_data_dir`, `python -m piper.download_voices` fetches it.
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


def warm_up() -> None:
    """Download/load both voices ahead of the first caller."""
    if not settings.tts_enabled:
        return
    for lang in ("en", "hi"):
        try:
            _load(lang)
        except Exception:
            log.exception("Could not load Piper voice for '%s' — TTS for that language will fail over to the browser", lang)


def synthesize(text: str) -> bytes:
    """Render `text` to WAV bytes with the language-appropriate voice.

    Raises on failure — callers should treat that as "no server audio for
    this sentence" and let the frontend fall back to speechSynthesis rather
    than dropping the turn.
    """
    lang = detect_voice_language(text)
    voice = _load(lang)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buf.getvalue()
