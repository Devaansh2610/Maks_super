"""Always-on wake-phrase detection using Vosk (offline, free, lightweight).

Runs a small streaming recognizer continuously and fuzzy-matches its running
transcript against the configured wake phrase, so an exact match isn't
required — near-misses from imperfect ASR ("daddy's home" heard as "daddys
home") still fire.

VAD-gated: earlier versions fed every single audio frame into Vosk with no
filtering at all, which meant background noise (TV, hum, a door closing)
that Vosk's ASR happened to mis-transcribe into something wake-phrase-shaped
could trigger a false wake. Now a match is only ever considered on frames
webrtcvad actually classifies as speech — ambient noise on a silent frame
can't trigger anything, no matter what Vosk thinks it heard.
"""

from __future__ import annotations

import json
import queue
import threading

import sounddevice as sd
import vosk
import webrtcvad
from rapidfuzz import fuzz

from maks.settings import settings
from maks.voice.audio_io import FRAME_SAMPLES, SAMPLE_RATE, mic_device

vosk.SetLogLevel(-1)

_model: vosk.Model | None = None


def _get_model() -> vosk.Model:
    global _model
    if _model is None:
        _model = vosk.Model(settings.vosk_model_path)
    return _model


def wait_for_wake_word(stop_event: threading.Event | None = None) -> bool:
    """Blocks until the wake phrase is heard; returns True in that case, or
    False if `stop_event` was set first (used for clean shutdown).
    """
    recognizer = vosk.KaldiRecognizer(_get_model(), SAMPLE_RATE)
    vad = webrtcvad.Vad(settings.vad_aggressiveness)
    q: "queue.Queue[bytes]" = queue.Queue()
    wake_phrase = settings.wake_phrase.lower()
    threshold = settings.wake_match_threshold

    def _callback(indata, frames, time_info, status):  # noqa: ANN001, ARG001
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        device=mic_device(),
        callback=_callback,
    ):
        while True:
            if stop_event is not None and stop_event.is_set():
                return False

            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Vosk still sees every frame (it needs continuous audio context
            # to transcribe accurately) -- only the match check below is
            # gated by VAD, so a noise-only frame can never trigger a wake
            # even if Vosk briefly mis-hears something phrase-shaped in it.
            is_speech = vad.is_speech(data, SAMPLE_RATE)

            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
            else:
                text = json.loads(recognizer.PartialResult()).get("partial", "")

            if is_speech and text and fuzz.partial_ratio(wake_phrase, text.lower()) >= threshold:
                recognizer.Reset()
                return True
