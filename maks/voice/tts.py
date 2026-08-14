"""Text-to-speech via the Fish Audio cloud API (s2.1-pro-free, a specific
voice reference). Requests WAV output directly so it drops straight into the
same local-playback path Piper used to use — no new audio-decoding
dependency needed.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import httpx

from maks.settings import settings
from maks.voice.audio_io import play_wav_file

_TTS_URL = "https://api.fish.audio/v1/tts"

# speak() can now be called concurrently — the voice loop's own greeting/
# reply calls, plus fire-and-forget "working on it" announcements fired from
# inside the async graph (see maks/agents/_common.py's announce_delegation)
# — and sounddevice's playback isn't safe to overlap (a second sd.play()
# call while one is still running can cut the first off). This lock just
# serializes actual synthesis+playback so concurrent callers queue up
# instead of garbling each other.
_speak_lock = threading.Lock()


def speak(text: str) -> None:
    """Synthesize `text` with Fish Audio and play it through the default
    speaker. Logs and returns on failure rather than crashing the assistant
    loop — same contract Piper had. Safe to call from multiple threads at
    once (blocks until any in-progress speech finishes first).
    """
    text = text.strip()
    if not text:
        return

    with _speak_lock:
        _speak_now(text)


def _speak_now(text: str) -> None:
    try:
        resp = httpx.post(
            _TTS_URL,
            headers={
                "Authorization": f"Bearer {settings.fish_audio_api_key}",
                "model": settings.fish_audio_model,
            },
            json={
                "text": text,
                "reference_id": settings.fish_audio_voice_id,
                "format": "wav",
                "sample_rate": 44100,
            },
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[tts] Fish Audio request failed: {exc}")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(resp.content)
        out_path = Path(tmp.name)

    try:
        play_wav_file(str(out_path))
    finally:
        out_path.unlink(missing_ok=True)
