"""Speech-to-text: OpenAI gpt-4o-mini-transcribe (primary), Groq-hosted Whisper
(fast/cheap fallback), then local faster-whisper (offline last resort).

The browser sends one complete WebM/Opus blob per utterance, which all three
paths can decode directly.

Whisper models — which is what both Groq and the local fallback actually
run — are notorious for hallucinating short, plausible-sounding phrases
(video-caption sign-offs like "Thank you" or "Halo") when fed near-silent or
noisy audio instead of returning nothing. Those two paths ask for
segment-level confidence (no_speech_prob / avg_logprob) and drop segments
that look like noise rather than speech, then run the survivors past a small
blocklist of known hallucination phrases as a backstop. gpt-4o-mini-transcribe
is a different (non-Whisper) architecture that's far less prone to this in
the first place and doesn't expose the same segment metadata, so it gets a
lighter, audio-length-based version of the same blocklist check instead.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Any

from openai import AsyncOpenAI

from .config import settings

log = logging.getLogger("receptionist.stt")

_local_model = None
_local_lock = threading.Lock()

_groq_client: AsyncOpenAI | None = None
_openai_stt_client: AsyncOpenAI | None = None

# The openai SDK does `base_url = base_url or os.environ.get("OPENAI_BASE_URL")`
# internally when base_url isn't passed — and os.environ.get returns "" (not
# None) for a blank-but-set var, which the SDK then treats as a real
# override instead of falling through to its own default. Pinning this
# explicitly is the only way to keep this client on real OpenAI regardless
# of whatever OPENAI_BASE_URL the chat model's client is using.
_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Segments this unconfident are almost always silence/noise, not speech.
_NO_SPEECH_PROB_MAX = 0.6
_AVG_LOGPROB_MIN = -1.0

# Below this many bytes of WebM/Opus, treat a blocklisted phrase as suspect
# even without segment-level confidence to check (roughly under a second).
_SHORT_AUDIO_BYTES = 12000

# Known Whisper hallucinations — only treated as such when confidence (or,
# for the OpenAI path, audio length) is also borderline, so a caller
# genuinely saying "thank you" or "bye" at the end of a call still comes
# through.
_HALLUCINATION_PHRASES = frozenset({
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you very much", "thanks so much for watching",
    "please subscribe", "like and subscribe", "subscribe to my channel",
    "share and subscribe", "bye", "bye bye", "goodbye", "okay bye",
    "halo", "hello", "hello hello", "you're welcome", "see you next time",
    "captions by", "transcription by castingwords", "amara.org",
})


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


def _get_openai_stt_client() -> AsyncOpenAI | None:
    global _openai_stt_client
    if not settings.openai_stt_api_key:
        return None
    if _openai_stt_client is None:
        _openai_stt_client = AsyncOpenAI(
            api_key=settings.openai_stt_api_key, base_url=_OPENAI_BASE_URL
        )
    return _openai_stt_client


def warm_up() -> None:
    """Load the local fallback model ahead of the first caller.

    Loaded regardless of which provider is primary, so an outage upstream
    degrades to an already-warm local model rather than a cold one.
    """
    _load_local()


def _looks_like_hallucination(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    normalized = text.strip().lower().strip(" .!?,")
    if normalized not in _HALLUCINATION_PHRASES:
        return False
    return no_speech_prob > 0.3 or avg_logprob < -0.5


def _join_segments(segments: list[tuple[str, float, float]]) -> str:
    """Drop low-confidence segments, then blocklist-check what's left."""
    kept = [
        text for text, no_speech_prob, avg_logprob in segments
        if no_speech_prob < _NO_SPEECH_PROB_MAX and avg_logprob > _AVG_LOGPROB_MIN
    ]
    text = " ".join(t.strip() for t in kept if t.strip()).strip()
    if not text:
        return ""
    worst_no_speech = max((n for _, n, _ in segments), default=0.0)
    worst_logprob = min((a for _, _, a in segments), default=0.0)
    if _looks_like_hallucination(text, worst_no_speech, worst_logprob):
        return ""
    return text


def _transcribe_local_sync(audio: bytes) -> str:
    model = _load_local()
    segments, _info = model.transcribe(
        io.BytesIO(audio),
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        language=settings.stt_language or None,
    )
    parsed = [(seg.text, seg.no_speech_prob, seg.avg_logprob) for seg in segments]
    return _join_segments(parsed)


async def _transcribe_groq(audio: bytes) -> str:
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("No GROQ_API_KEY configured")
    buf = io.BytesIO(audio)
    buf.name = "utterance.webm"
    kwargs: dict[str, Any] = {
        "model": settings.groq_stt_model,
        "file": buf,
        "response_format": "verbose_json",
    }
    if settings.stt_language:
        kwargs["language"] = settings.stt_language
    response = await client.audio.transcriptions.create(**kwargs)
    segments = getattr(response, "segments", None) or []
    if not segments:
        # Some models/providers omit segment-level detail — fall back to the
        # plain transcript rather than discarding a real utterance.
        return (getattr(response, "text", "") or "").strip()
    parsed = [(seg.text, seg.no_speech_prob, seg.avg_logprob) for seg in segments]
    return _join_segments(parsed)


async def _transcribe_openai(audio: bytes) -> str:
    client = _get_openai_stt_client()
    if client is None:
        raise RuntimeError("No OPENAI_STT_API_KEY configured")
    buf = io.BytesIO(audio)
    buf.name = "utterance.webm"
    # gpt-4o-mini-transcribe only supports response_format="json" — no
    # verbose_json/segments, unlike the Whisper-based paths above.
    kwargs: dict[str, Any] = {"model": settings.openai_stt_model, "file": buf}
    if settings.stt_language:
        kwargs["language"] = settings.stt_language
    response = await client.audio.transcriptions.create(**kwargs)
    text = (response.text or "").strip()
    if not text:
        return ""
    normalized = text.lower().strip(" .!?,")
    if normalized in _HALLUCINATION_PHRASES and len(audio) < _SHORT_AUDIO_BYTES:
        return ""
    return text


# Tried in order for STT_PROVIDER="openai"; STT_PROVIDER="groq" skips
# straight to Groq (e.g. to avoid paying for OpenAI STT at all).
_CHAIN: dict[str, list[str]] = {
    "openai": ["openai", "groq"],
    "groq": ["groq"],
    "local": [],
}
_PROVIDER_FNS = {"openai": _transcribe_openai, "groq": _transcribe_groq}


async def transcribe(audio: bytes) -> str:
    """Transcribe one utterance. Returns "" when nothing intelligible was said."""
    if len(audio) < 2000:  # a fraction of a second of silence
        return ""

    for name in _CHAIN.get(settings.stt_provider, []):
        try:
            return await _PROVIDER_FNS[name](audio)
        except Exception:
            log.exception("%s transcription failed — trying the next provider", name)

    return await asyncio.to_thread(_transcribe_local_sync, audio)
