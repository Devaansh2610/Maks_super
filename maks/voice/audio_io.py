"""Microphone capture + speaker playback helpers (sounddevice + webrtcvad).

Everything here works in mono 16kHz int16 PCM — the format both Vosk and
faster-whisper want, and what webrtcvad requires for voice-activity framing.
"""

from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

from maks.settings import settings

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples = 960 bytes (int16)


def mic_device() -> str | int | None:
    return settings.mic_device or None


def speaker_device() -> str | int | None:
    return settings.speaker_device or None


def record_until_silence(
    max_seconds: float = 15.0,
    silence_seconds: float | None = None,
    min_speech_seconds: float = 0.3,
) -> np.ndarray:
    """Record from the mic until `silence_seconds` of silence follows at
    least `min_speech_seconds` of detected speech, or `max_seconds` elapses.
    Returns int16 mono samples (empty array if nothing was ever recorded).

    `silence_seconds` defaults to settings.command_silence_seconds (tunable
    via COMMAND_SILENCE_SECONDS in .env) rather than a fixed constant, so a
    noisy room that keeps tripping VAD as "still speech" can be tuned down
    without a code change.
    """
    if silence_seconds is None:
        silence_seconds = settings.command_silence_seconds

    vad = webrtcvad.Vad(settings.vad_aggressiveness)
    q: "queue.Queue[bytes]" = queue.Queue()

    def _callback(indata, frames, time_info, status):  # noqa: ANN001, ARG001
        q.put(bytes(indata))

    frames_buf: list[bytes] = []
    speech_ms = 0
    silence_ms = 0
    total_ms = 0
    started_speech = False

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        device=mic_device(),
        callback=_callback,
    ):
        while total_ms < max_seconds * 1000:
            chunk = q.get()
            frames_buf.append(chunk)
            total_ms += FRAME_MS

            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            if is_speech:
                speech_ms += FRAME_MS
                silence_ms = 0
                if speech_ms >= min_speech_seconds * 1000:
                    started_speech = True
            elif started_speech:
                silence_ms += FRAME_MS
                if silence_ms >= silence_seconds * 1000:
                    break

    raw = b"".join(frames_buf)
    return np.frombuffer(raw, dtype=np.int16)


def play_wav_file(path: str) -> None:
    data, samplerate = sf.read(path, dtype="int16")
    sd.play(data, samplerate=samplerate, device=speaker_device())
    sd.wait()
