"""Command transcription: VAD-based recording + faster-whisper (offline,
CPU-friendly via CTranslate2 int8 quantization)."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from maks.settings import settings
from maks.voice.audio_io import record_until_silence

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def listen_and_transcribe(max_seconds: float = 15.0) -> str:
    """Record a spoken command (until the user stops talking) and transcribe it."""
    audio = record_until_silence(max_seconds=max_seconds)
    if audio.size == 0:
        return ""

    audio_float = audio.astype(np.float32) / 32768.0
    segments, _info = _get_model().transcribe(audio_float, language="en", vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()
