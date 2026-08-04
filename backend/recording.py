"""Per-call audio recording.

Both sides of a call arrive in different containers — the caller's audio is
whatever WebM/Opus blob the browser's MediaRecorder produced, the
assistant's is WAV from Piper or OpenAI TTS — so each piece is decoded and
resampled to a common mono 16kHz PCM stream via PyAV (already a transitive
dependency through faster-whisper) as it arrives, then concatenated in call
order and written out as one WAV file when the call ends.
"""

from __future__ import annotations

import logging
import wave
from io import BytesIO
from pathlib import Path

import av

from .config import settings

log = logging.getLogger("receptionist.recording")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


def _plane_bytes(frame: av.AudioFrame) -> bytes:
    """The frame's actual PCM bytes, not PyAV's (larger, alignment-padded) plane buffer.

    `bytes(frame.planes[0])` grabs the whole underlying buffer, which ffmpeg
    allocates padded to an alignment boundary — the tail is uninitialized
    garbage, not silence. Left in, it injects a burst of noise at every
    single frame boundary (every ~20-85ms) throughout the recording, audible
    as a constant ticking. Slicing to `frame.samples` is the fix.
    """
    valid_bytes = frame.samples * SAMPLE_WIDTH * CHANNELS
    return bytes(frame.planes[0])[:valid_bytes]


def _decode_to_pcm(data: bytes) -> bytes:
    """Decode any container PyAV understands to mono 16kHz 16-bit PCM bytes."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm = bytearray()
    with av.open(BytesIO(data)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                pcm += _plane_bytes(resampled)
    for resampled in resampler.resample(None):  # flush
        pcm += _plane_bytes(resampled)
    return bytes(pcm)


class CallRecorder:
    """Accumulates one call's audio in order; `save()` once at hang-up."""

    def __init__(self, call_id: int) -> None:
        self.call_id = call_id
        self._pcm = bytearray()

    def add(self, audio_bytes: bytes) -> None:
        """Decode and append a segment (caller utterance or spoken reply)."""
        if not audio_bytes:
            return
        try:
            self._pcm += _decode_to_pcm(audio_bytes)
        except Exception:
            log.exception(
                "Call %s: could not decode an audio segment for the recording — "
                "skipping it, the rest of the call is unaffected", self.call_id
            )

    @property
    def duration_seconds(self) -> float:
        return len(self._pcm) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)

    def save(self) -> Path | None:
        """Write the accumulated audio to disk. Returns None if nothing was recorded."""
        if not self._pcm:
            return None
        path = settings.recordings_dir / f"{self.call_id}.wav"
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(bytes(self._pcm))
        return path
