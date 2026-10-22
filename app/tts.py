"""Server-side text-to-speech using Microsoft Edge's free neural voices (edge-tts).

No API key needed, but the server needs internet access. Browsers on Linux often have
no Hindi voice, so the call console plays this audio and only falls back to the
browser's own speech engine if this fails.
"""
import os

import edge_tts

VOICES = {
    "hi-IN": os.environ.get("TTS_VOICE_HI", "hi-IN-SwaraNeural"),
    "en-IN": os.environ.get("TTS_VOICE_EN", "en-IN-NeerjaNeural"),
}

_cache: dict[tuple[str, str], bytes] = {}
_CACHE_MAX = 200


async def synthesize(text: str, lang: str = "hi-IN") -> bytes:
    voice = VOICES.get(lang, VOICES["hi-IN"])
    key = (voice, text)
    if key in _cache:
        return _cache[key]
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("TTS returned no audio")
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = bytes(audio)
    return _cache[key]
