"""
voice.py — speech-to-text via ElevenLabs Scribe.

WHY ElevenLabs Scribe:
- One voice vendor for the whole pipeline. I already use ElevenLabs for
  TTS in other side projects, and having STT and TTS in the same place
  keeps the credentials surface small and the production story simple.
- Scribe handles Norwegian competently (the model auto-detects, but we
  pass an ISO language hint for stability on short clips).

For production at Gjensidige this would likely become a managed Azure
Speech endpoint for data-residency reasons — the function signature
(`transcribe(path) -> str`) stays the same so swapping is a one-file change.
"""

from pathlib import Path

from elevenlabs.client import ElevenLabs


def transcribe(audio_path: str, language: str = "nor") -> str:
    """Transcribe an audio file to text. Returns the plain transcript."""
    client = ElevenLabs()  # reads ELEVENLABS_API_KEY from env
    with Path(audio_path).open("rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            language_code=language,  # ISO 639-3 — "nor" = Norwegian
        )
    # The SDK returns an object with .text on it
    return getattr(result, "text", str(result))
