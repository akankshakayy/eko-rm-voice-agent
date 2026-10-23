"""Server-side speech-to-text using Groq's free Whisper API (same GROQ_API_KEY as the LLM).

Lets the mic work in any browser. Without a Groq key the call console falls back to
the browser's own speech recognition (Chrome / Edge only).
"""
import os
import shutil
import subprocess

MODEL = os.environ.get("STT_MODEL", "whisper-large-v3-turbo")
_client = None


def available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


# What Groq's Whisper accepts; anything else (WhatsApp .aac, phone .amr, .3gp) is converted first
SUPPORTED = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "opus", "wav", "webm"}


def to_supported(audio: bytes, filename: str) -> tuple[bytes, str]:
    """Convert audio Whisper can't take into 16 kHz mono FLAC with ffmpeg (lossless, small)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in SUPPORTED:
        return audio, filename
    if not shutil.which("ffmpeg"):
        raise ValueError(f"Can't transcribe .{ext or '?'} audio without ffmpeg installed")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                          "-f", "flac", "pipe:1"], input=audio, capture_output=True, timeout=120)
    if out.returncode != 0 or not out.stdout:
        raise ValueError(f"Couldn't read this audio file: {out.stderr.decode(errors='replace')[:200]}")
    return out.stdout, filename.rsplit(".", 1)[0] + ".flac"


def _groq():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1",
                         timeout=30, max_retries=1)
    return _client


# Steers Whisper towards Hinglish in Roman script, which is what the agent expects
PROMPT = ("Eko CSP support call. DMT, AePS, wallet, commission, KYC, refund, transaction ID, UTR. "
          "हाँ जी, मेरा DMT transfer fail हो गया। My money transfer failed.")


def transcribe(audio: bytes, filename: str, lang: str = "hi-IN") -> str:
    audio, filename = to_supported(audio, filename)
    # "auto" = let Whisper detect it (used while the caller is still choosing Hindi or English)
    extra = {} if lang == "auto" else {"language": lang.split("-")[0]}
    r = _groq().audio.transcriptions.create(model=MODEL, file=(filename, audio), temperature=0, prompt=PROMPT, **extra)
    return (r.text or "").strip()


def transcribe_segments(audio: bytes, filename: str, lang: str = "hi", model: str | None = None) -> list[dict]:
    """A whole call, as timed segments: [{"start": 0.0, "end": 4.2, "text": "..."}]. Whisper doesn't say who spoke."""
    return transcribe_detailed(audio, filename, lang, model)[0]


def transcribe_detailed(audio: bytes, filename: str, lang: str = "hi", model: str | None = None,
                        words: bool = False, prompt: str | None = PROMPT) -> tuple[list[dict], list[dict]]:
    """A whole call as timed segments, plus (words=True) every word with its own time, for matching to voices.
    prompt=None sends no hint: on long recordings Whisper sometimes copies a hint sentence into the transcript."""
    audio, filename = to_supported(audio, filename)
    extra = {} if lang == "auto" else {"language": lang.split("-")[0]}
    if prompt:
        extra["prompt"] = prompt
    r = _groq().with_options(timeout=300).audio.transcriptions.create(
        model=model or MODEL, file=(filename, audio), temperature=0, response_format="verbose_json",
        timestamp_granularities=["word", "segment"] if words else ["segment"], **extra)
    get = lambda s, k: s.get(k) if isinstance(s, dict) else getattr(s, k, None)  # noqa: E731
    segs = [{"start": round(float(get(s, "start") or 0), 2), "end": round(float(get(s, "end") or 0), 2),
             "text": (get(s, "text") or "").strip()} for s in (getattr(r, "segments", None) or [])
            if (get(s, "text") or "").strip()]
    ws = [{"start": round(float(get(w, "start") or 0), 2), "end": round(float(get(w, "end") or 0), 2),
           "word": (get(w, "word") or "").strip()} for w in (getattr(r, "words", None) or []) if (get(w, "word") or "").strip()]
    return segs, ws
