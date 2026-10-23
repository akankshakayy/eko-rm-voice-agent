"""Pilot: real RM-CSP call recordings, uploaded every day from the RMs' phones (Eko-issued SIMs).

The uploader app (uploader/) sends the dialer's own recording of every call on the Eko SIM to
POST /api/uploader/recordings a few minutes after each call. Every new file is processed as soon as it
arrives (or once a day, if RECORDINGS_DAILY_AT is set, e.g. 21:00 IST):
  1. transcribed with Whisper on Groq, as timed segments
  2. masked: Aadhaar, PAN, card, account, phone and OTP/PIN numbers are cut from the text before it is
     stored or sent to the LLM (regex only: numbers spoken as Hindi words get through)
  3. reviewed by the LLM: who said each line (RM or CSP), issue type, details, CSP mood, how it ended,
     how the RM handled it, and a summary
Audio and results stay in data/ (git-ignored). `export` cuts the CSP's opening complaint from each finished
call into draft scorecard cases, for a person to check before they go into evals/cases.csv.

    python -m app.recordings token --rm 4 --label "Shiwani's phone"  # upload token for one RM's phone
    python -m app.recordings token --label "pilot phones"            # shared: each RM picks their name in the app
    python -m app.recordings tokens | revoke 3
    python -m app.recordings process                                # process everything new right now
    python -m app.recordings list
    python -m app.recordings export                                 # draft cases -> evals/candidates.csv
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import db, diarize, integrations, llm_engine, report, sarvam, stt
from .taxonomy import CATEGORIES, SLOT_LABELS

log = logging.getLogger("eko.recordings")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 100 * 1024 * 1024
MAX_ATTEMPTS = 3                 # a recording that fails this many runs is marked failed
MAX_PROMPT_CHARS = 9000          # ~3k tokens: fits a Groq free-tier minute with room for the answer
RATE_LIMIT_RETRIES = 3
DAILY_AT = os.environ.get("RECORDINGS_DAILY_AT", "").strip()   # "" = process each recording as it arrives
RETRY_AFTER_S = 600              # a failed recording waits 10 min, then 20, before its next try
TZ = ZoneInfo(os.environ.get("RECORDINGS_TZ", "Asia/Kolkata"))
LANG = os.environ.get("RECORDINGS_LANG", "hi")
# Recorded calls aren't live, so they get Whisper's most accurate model (the live agent uses the faster turbo one)
STT_MODEL = os.environ.get("RECORDINGS_STT_MODEL", "whisper-large-v3")
# A style-only hint: without one, Whisper writes English words in Devanagari ("वेबसाइट") with no punctuation, and
# mishears more. It must carry no facts: the live agent's hint ("मेरा DMT transfer fail हो गया") once turned up as
# a line in a meeting that never mentioned a DMT. Lines that repeat the hint are dropped (_drop_leaks) as well.
STT_PROMPT = os.environ.get("RECORDINGS_STT_PROMPT", "हाँ जी, ठीक है। तो ये website का project है, उसमें docs folder है।") or None
AUDIO_EXTS = {"m4a", "mp3", "aac", "amr", "3gp", "ogg", "opus", "wav", "flac", "mp4", "webm"}
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
SPEAKERS = ["rm", "csp", "other", "unknown"]
RESOLUTIONS = ["resolved_on_call", "follow_up_promised", "escalated", "no_issue", "unclear"]


class RateLimited(Exception):
    """A free-tier limit was hit. The recording stays new and is tried again later."""
    def __init__(self, message: str, wait: float = 60):
        super().__init__(message)
        self.wait = wait


def storage_dir() -> str:
    return os.environ.get("RECORDINGS_DIR") or os.path.join(os.path.dirname(os.path.abspath(db.DB_PATH)), "recordings")


# ---------------------------------------------------------------- upload tokens (one per RM phone)
def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


TOKEN_COLS = """t.id, t.rm_id, r.name AS rm_name, t.label, t.created_by, t.created_at, t.last_used_at, t.revoked_at,
                (SELECT COUNT(*) FROM recordings WHERE token_id=t.id) AS uploads,
                (SELECT MAX(uploaded_at) FROM recordings WHERE token_id=t.id) AS last_upload_at"""


def create_token(rm_id: int | None, label: str = "", created_by: str = "") -> dict:
    """The token itself is returned only here; the database keeps just its hash.
    rm_id=None makes a shared token: each RM picks their own name in the app, and it goes with every upload."""
    if rm_id is not None and not db.q("SELECT id FROM rms WHERE id=?", (rm_id,), one=True):
        raise ValueError("No such RM")
    token = "ekoup_" + secrets.token_urlsafe(24)
    tid = db.x("INSERT INTO upload_tokens(token_hash,rm_id,label,created_by,created_at) VALUES(?,?,?,?,?)",
               (_h(token), rm_id, (label or "").strip()[:80], created_by, time.time()))
    return {**db.q(f"SELECT {TOKEN_COLS} FROM upload_tokens t LEFT JOIN rms r ON r.id=t.rm_id WHERE t.id=?",
                   (tid,), one=True), "token": token}


def list_tokens() -> list[dict]:
    return db.q(f"SELECT {TOKEN_COLS} FROM upload_tokens t LEFT JOIN rms r ON r.id=t.rm_id ORDER BY t.id DESC")


def revoke_token(token_id: int) -> bool:
    if not db.q("SELECT id FROM upload_tokens WHERE id=? AND revoked_at IS NULL", (token_id,), one=True):
        return False
    db.update("upload_tokens", token_id, revoked_at=time.time())
    return True


def device_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    dev = db.q("""SELECT t.*, r.name AS rm_name FROM upload_tokens t LEFT JOIN rms r ON r.id=t.rm_id
                  WHERE t.token_hash=? AND t.revoked_at IS NULL""", (_h(token),), one=True)
    if dev:
        db.update("upload_tokens", dev["id"], last_used_at=time.time())
    return dev


def rms_for_app() -> list[dict]:
    """The RMs the app offers to choose from: id and first name only (no contact details)."""
    return [{"id": r["id"], "name": r["name"]} for r in db.q("SELECT id, name FROM rms ORDER BY name")]


def uploader_rm(device: dict, chosen: int | None) -> int:
    """Which RM an upload belongs to. A personal token decides it (and refuses another RM's name); a shared token
    takes the RM chosen in the app."""
    if device.get("rm_id"):
        if chosen and int(chosen) != device["rm_id"]:
            raise ValueError(f"This phone's upload token belongs to {device['rm_name']}. Ask the admin for your own.")
        return device["rm_id"]
    if not chosen or not db.q("SELECT id FROM rms WHERE id=?", (int(chosen),), one=True):
        raise ValueError("Choose your name (RM) in the app's setup, then try again")
    return int(chosen)


# ---------------------------------------------------------------- receiving files
def save_upload(device: dict, data: bytes, filename: str, started_at_ms: int | None = None,
                duration_sec: float | None = None, direction: str = "unknown", other_number: str = "",
                device_id: str = "") -> dict:
    """Store one recording from a phone. The same file sent twice is stored once (matched by SHA-256)."""
    if not data:
        raise ValueError("Empty file")
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    if ext not in AUDIO_EXTS:
        raise ValueError(f"Not a call recording: .{ext or '?'}")
    sha = hashlib.sha256(data).hexdigest()
    old = db.q("SELECT id FROM recordings WHERE sha256=?", (sha,), one=True)
    if old:
        return {"id": old["id"], "duplicate": True}
    started = started_at_ms / 1000 if started_at_ms else None
    day = datetime.fromtimestamp(started or time.time(), TZ).strftime("%Y-%m-%d")
    rel = os.path.join(f"rm{device['rm_id']}" if device.get("rm_id") else "no-rm", day, f"{sha[:16]}.{ext}")
    path = os.path.join(storage_dir(), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if duration_sec is None:
        duration_sec = _probe_duration(path)
    digits = re.sub(r"\D", "", other_number or "")[-10:]
    csp = db.q("SELECT id FROM csps WHERE phone=?", (digits,), one=True) if digits else None
    try:
        rid = db.x("""INSERT INTO recordings(sha256,rm_id,token_id,device_id,file_path,orig_name,size_bytes,started_at,
                      duration_sec,direction,other_number,csp_id,uploaded_at,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'new')""",
                   (sha, device["rm_id"], device["id"], (device_id or "")[:64], rel, (filename or "")[:200], len(data),
                    started, duration_sec, direction if direction in ("incoming", "outgoing") else "unknown",
                    digits or None, csp["id"] if csp else None, time.time()))
    except sqlite3.IntegrityError:          # the same file arrived twice at once
        return {"id": db.q("SELECT id FROM recordings WHERE sha256=?", (sha,), one=True)["id"], "duplicate": True}
    wake()
    return {"id": rid, "duplicate": False}


def _probe_duration(path: str) -> float | None:
    """Length in seconds, for files added by hand (phones send it with the upload)."""
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=60)
    try:
        return round(float(out.stdout.strip()), 1)
    except ValueError:
        return None


def audio_file(rec: dict) -> str:
    return os.path.join(storage_dir(), rec["file_path"])


def reprocess(rec_id: int) -> bool:
    """Run a recording through transcription and review again (after a failure, or a better model or prompt)."""
    if not db.q("SELECT id FROM recordings WHERE id=?", (rec_id,), one=True):
        return False
    db.update("recordings", rec_id, status="new", attempts=0, error=None, next_try_at=None)
    wake()
    return True


def delete_recording(rec_id: int) -> bool:
    """Remove a recording and its audio file for good."""
    r = db.q("SELECT id, file_path FROM recordings WHERE id=?", (rec_id,), one=True)
    if not r:
        return False
    try:
        os.remove(audio_file(r))
    except FileNotFoundError:
        pass
    db.x("DELETE FROM recordings WHERE id=?", (rec_id,))
    return True


# ---------------------------------------------------------------- masking personal numbers
_ID_WORDS = re.compile(r"(?i)txn|transaction|utr|\bref|\bid\b|आईडी|ट्रांजैक्शन")
_MASKS = [
    (re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b|\b\d{16}\b"), "[CARD]"),
    (re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"), "[AADHAAR]"),      # Aadhaar is read out in fours; UTRs aren't
    (re.compile(r"(?i)((?:aadha?a?r|adhar|आधार)\D{0,25})\d{12}\b"), r"\1[AADHAAR]"),
    (re.compile(r"(?i)\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"(?i)((?:otp|m?pin|password|ओटीपी|पिन|पासवर्ड)\D{0,20})\d{4,8}\b"), r"\1[OTP]"),
    (re.compile(r"(?i)((?:account|a/c|khata|खाता|अकाउंट)\D{0,25})\d{9,18}\b"), r"\1[ACCOUNT]"),
]
_PHONE = re.compile(r"(?<!\d)(?:\+?91[ -]?|0)?[6-9]\d{9}(?!\d)")


def mask(text: str) -> str:
    """Cut personal numbers from a transcript line. Transaction IDs, UTRs and amounts are kept:
    they are what the agent has to learn to catch."""
    text = (text or "").translate(DEVANAGARI_DIGITS)
    for rx, repl in _MASKS:
        text = rx.sub(repl, text)
    return _PHONE.sub(lambda m: m.group(0) if _ID_WORDS.search(m.string[max(0, m.start() - 25):m.start()])
                      else "[PHONE]", text)


# ---------------------------------------------------------------- reviewing a call with the LLM
REVIEW_TOOL = {
    "name": "call_review",
    "description": "Return who spoke each transcript line and what the call was about.",
    "parameters": {
        "type": "object",
        "properties": {
            "speakers": {"type": "array", "items": {"type": "string", "enum": SPEAKERS},
                         "description": "One entry per numbered line, in order: who said it."},
            "voice_roles": {"type": "object",
                            "properties": {v: {"type": "string", "enum": SPEAKERS} for v in "ABCD"},
                            "description": "When lines carry voice labels: who each voice is, e.g. {\"A\": \"rm\", "
                                           "\"B\": \"csp\"}."},
            "is_csp_call": {"type": "boolean",
                            "description": "False if this is not an RM-CSP work conversation (wrong number, "
                                           "voicemail, personal call, no real conversation)."},
            "language": {"type": "string", "enum": ["hi", "hinglish", "en", "other"]},
            "issue_segment": {"type": "integer",
                              "description": "Line number where the CSP first describes their main issue; -1 if none."},
            "intent": {"type": "string", "enum": llm_engine.INTENTS},
            "intent_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "slots": {"type": "object", "properties": {k: {"type": "string", "description": v}
                                                       for k, v in SLOT_LABELS.items()},
                      "description": "Issue details stated anywhere in the call. Leave out what wasn't said."},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "frustrated", "angry"],
                          "description": "The CSP's mood over the call."},
            "resolution": {"type": "string", "enum": RESOLUTIONS},
            "rm_handling": {"type": "string",
                            "description": "1-3 short English sentences: what the RM checked, explained or "
                                           "promised. Used to teach the voice agent how RMs solve this issue."},
            "summary": {"type": "string", "description": "1-2 English sentences: what the call was about."},
        },
        "required": ["speakers", "is_csp_call", "language", "issue_segment", "intent", "intent_confidence",
                     "sentiment", "resolution", "rm_handling", "summary"],
    },
}


def _review_prompt() -> str:
    cats = "\n".join(f"- {k}: {v['label']}" for k, v in CATEGORIES.items())
    return f"""You review recorded phone calls of Eko Relationship Managers (RMs). On most calls the RM talks to a
CSP: a retailer who offers Eko services (DMT money transfer, AePS cash withdrawal, BBPS bills, recharge) at
their shop. The transcript comes from speech-to-text: it has mistakes and does not say who spoke. Personal
numbers are already masked, e.g. [PHONE], [AADHAAR].

For every numbered line decide who said it: "rm" (the Eko RM: greets the CSP by name, asks for details,
checks, explains, promises a follow-up), "csp" (the retailer: reports problems, asks about money, commission,
their ID), "other" (anyone else, e.g. a customer at the shop) or "unknown". Speakers usually alternate, but one
person often speaks several lines in a row.
When lines carry a voice label, e.g. "(voice A)", the voices were separated from the sound itself, so every line
with the same voice is the same person. Then fill voice_roles (which person each voice is), judging from
everything that voice says, and give each line its voice's role in speakers.
Then classify the call into one issue type:
{cats}
- unknown: none of these, or no issue
Use the call_review tool. Write summary and rm_handling in English."""


def _review_input(segs: list[dict]) -> str:
    lines, size = [], 0
    for i, s in enumerate(segs):
        voice = f" (voice {s['voice']})" if s.get("voice") else ""
        line = f"[{i}] {int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}{voice} {s['text']}"
        if size + len(line) > MAX_PROMPT_CHARS:
            lines.append(f"(transcript cut here: lines {i}-{len(segs) - 1} not shown, mark them unknown)")
            break
        lines.append(line)
        size += len(line) + 1
    return f"Transcript ({len(segs)} lines):\n" + "\n".join(lines)


def _ask(system: str, user: str, tool: dict) -> tuple[dict, str]:
    """One forced tool call on the configured LLM. Free-tier limits become RateLimited."""
    p = llm_engine.provider()
    if p == "claude":
        import anthropic
        try:
            resp = anthropic.Anthropic().messages.create(
                model=llm_engine.MODEL, max_tokens=4000, system=system, messages=[{"role": "user", "content": user}],
                tools=[{"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}],
                tool_choice={"type": "tool", "name": tool["name"]})
        except anthropic.RateLimitError as e:
            raise RateLimited("Claude rate limit") from e
        return dict(next(b for b in resp.content if b.type == "tool_use").input), llm_engine.MODEL
    from openai import APIStatusError, OpenAI, RateLimitError
    cfg = llm_engine.PROVIDERS[p]
    client = OpenAI(api_key=os.environ.get(cfg["key_env"]), base_url=cfg["base_url"], timeout=120, max_retries=0)
    models = [m.strip() for m in cfg["model"].split(",") if m.strip()]
    for i, model in enumerate(models):
        last = i == len(models) - 1
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=4000, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[{"type": "function", "function": tool}],
                tool_choice={"type": "function", "function": {"name": tool["name"]}})
        except RateLimitError as e:           # each model in the chain has its own free limit
            if last:
                wait = float(e.response.headers.get("retry-after", 60)) if e.response is not None else 60
                raise RateLimited(f"{cfg['label']} rate limit", wait) from e
            continue
        except APIStatusError as e:           # tool-call slips (400) or a call too long for this model (413)
            if last or e.status_code not in (400, 413):
                raise
            continue
        calls = resp.choices[0].message.tool_calls
        if calls:
            return json.loads(calls[0].function.arguments), model
        if last:
            raise ValueError("The model returned no call_review")
    raise ValueError("No model returned a call_review")


def review_call(segs: list[dict]) -> tuple[dict, str]:
    out, model = _ask(_review_prompt(), _review_input(segs), REVIEW_TOOL)
    speakers = [s if s in SPEAKERS else "unknown" for s in (out.get("speakers") or [])]
    speakers = (speakers + ["unknown"] * len(segs))[:len(segs)]
    roles = {v: r for v, r in (out.get("voice_roles") or {}).items() if r in SPEAKERS}
    # Voices come from the sound, so one voice is one person: its role decides every line it speaks
    out["speakers"] = [roles.get(s.get("voice"), sp) for s, sp in zip(segs, speakers)]
    out["slots"] = {k: llm_engine.normalize_id(v) if k in llm_engine.ID_SLOTS else str(v).strip()
                    for k, v in (out.get("slots") or {}).items() if k in SLOT_LABELS and str(v or "").strip()}
    try:
        i = int(out.get("issue_segment", -1))
    except (TypeError, ValueError):
        i = -1
    out["issue_segment"] = i if 0 <= i < len(segs) else -1
    if out.get("intent") not in llm_engine.INTENTS:
        out["intent"] = "unknown"
    if out.get("resolution") not in RESOLUTIONS:
        out["resolution"] = "unclear"
    return out, model


# ---------------------------------------------------------------- processing
def _compact(path: str) -> tuple[bytes, str]:
    """Any phone format -> 16 kHz mono Opus, so a long call stays under Whisper's 25 MB limit.
    Reads the file itself, not a pipe: phone .m4a files often keep their index at the end."""
    if not shutil.which("ffmpeg"):
        with open(path, "rb") as f:
            return f.read(), os.path.basename(path)
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000",
                          "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "pipe:1"], capture_output=True, timeout=600)
    if out.returncode != 0 or not out.stdout:
        raise ValueError(f"Couldn't read this audio file: {out.stderr.decode(errors='replace')[:200]}")
    return out.stdout, "call.ogg"


_SENTENCE_END = re.compile(r"(?<=[।?!])\s+|(?<=[A-Za-z\u0900-\u097F]{3}\.)\s+")   # not after "Rs." or "2.5"


def split_sentences(segs: list[dict]) -> list[dict]:
    """Whisper cuts audio at pauses, not at a change of speaker, so one segment often holds the RM's question and
    the CSP's answer. Split each segment into sentences (times shared out by length) so each can get its own speaker."""
    out = []
    for s in segs:
        parts = [p.strip() for p in _SENTENCE_END.split(s["text"]) if p.strip()]
        if len(parts) < 2:
            out.append(s)
            continue
        total, t = sum(len(p) for p in parts), s["start"]
        for p in parts:
            end = round(t + (s["end"] - s["start"]) * len(p) / total, 2)
            out.append({**s, "start": round(t, 2), "end": end, "text": p})
            t = end
    return out


def _is_rate_limit(e: Exception) -> bool:
    return type(e).__name__ == "RateLimitError" or getattr(e, "status_code", None) == 429


def _lines(r: dict, path: str, cache: dict) -> list[dict]:
    """Transcript lines, one speaker each: Sarvam's own speaker split; else Whisper's words matched to the local
    voice model; else Whisper's sentences."""
    if "lines" in cache:
        return split_sentences(cache["lines"])
    segs, words = cache["segments"], cache["words"]
    if words and diarize.available():
        try:
            if cache.get("voice_settings") != diarize.settings():
                t0 = time.perf_counter()
                cache.update(turns=diarize.diarize(path), voice_settings=diarize.settings(),
                             voices_s=round(time.perf_counter() - t0, 1), voices_at=time.time())
                db.update("recordings", r["id"], stt_json=cache)
            if cache["turns"]:
                return split_sentences(diarize.assign_words(diarize.punctuate(words, segs), cache["turns"]))
        except Exception:
            log.exception("Voice separation failed for %s; splitting by sentence instead", path)
    return split_sentences(segs)


PROMPT_UNSET = object()     # results saved before the prompt was recorded are redone


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _drop_leaks(segs: list[dict], words: list[dict], prompt: str | None) -> tuple[list[dict], list[dict]]:
    """Whisper sometimes repeats its hint over silence or noise. Drop every segment that mostly repeats it (and
    that segment's words), so a hint can never show up as something a person said."""
    if not prompt:
        return segs, words
    from difflib import SequenceMatcher
    p = _norm(prompt)
    leaked = [s for s in segs if (n := _norm(s["text"])) and
              (SequenceMatcher(None, n, p).ratio() >= 0.6 or (len(n) >= 12 and n in p))]
    if not leaked:
        return segs, words
    log.warning("Dropped %d transcript segment(s) that repeat the Whisper hint", len(leaked))
    inside = lambda w: any(s["start"] - 0.05 <= w["start"] < s["end"] + 0.05 for s in leaked)   # noqa: E731
    return [s for s in segs if s not in leaked], [w for w in words if not inside(w)]


def _transcribe(r: dict, path: str) -> dict:
    """The speech-to-text result, reused from an earlier run with the same settings when there is one.
    With SARVAM_API_KEY: Sarvam's transcript already split by speaker. Otherwise Whisper's segments and words."""
    cache = json.loads(r.get("stt_json") or "null") or {}
    if sarvam.available():
        if cache.get("provider") == sarvam.settings() and "lines" in cache:
            return cache
        t0 = time.perf_counter()
        try:
            lines = sarvam.transcribe(path)
        except sarvam.SarvamError as e:
            if e.status_code == 429:
                raise RateLimited("Sarvam rate limit", 120) from e
            raise
        cache = {"provider": sarvam.settings(), "stt": f"Sarvam {sarvam.MODEL} ({sarvam.MODE}), speakers by Sarvam",
                 "lines": lines, "transcribe_s": round(time.perf_counter() - t0, 1), "transcribe_at": time.time()}
        db.update("recordings", r["id"], stt_json=cache)
        return cache
    want_words = diarize.available()
    if cache.get("model") == STT_MODEL and cache.get("prompt", PROMPT_UNSET) == STT_PROMPT and (cache.get("words") or not want_words):
        return cache
    t0 = time.perf_counter()
    audio, name = _compact(path)
    try:
        segs, words = stt.transcribe_detailed(audio, name, LANG, STT_MODEL, words=want_words, prompt=STT_PROMPT)
    except Exception as e:
        if _is_rate_limit(e):
            raise RateLimited("Whisper rate limit", 300) from e
        raise
    segs, words = _drop_leaks(segs, words, STT_PROMPT)
    old = cache
    cache = {"model": STT_MODEL, "prompt": STT_PROMPT, "segments": segs, "words": words,
             "stt": f"Whisper {STT_MODEL}" + (", speakers by the local voice model" if want_words else ""),
             "transcribe_s": round(time.perf_counter() - t0, 1), "transcribe_at": time.time()}
    # Voice turns come from the audio, not the words: keep them when only the transcription is redone
    cache.update({k: old[k] for k in ("turns", "voice_settings", "voices_s", "voices_at") if k in old})
    db.update("recordings", r["id"], stt_json=cache)
    return cache


def process_one(r: dict):
    started, t0 = time.time(), time.perf_counter()
    path = os.path.join(storage_dir(), r["file_path"])
    cache = _transcribe(r, path)
    fresh_stt = cache.get("transcribe_at", 0) >= started
    t1 = time.perf_counter()
    segs = [{**s, "text": mask(s["text"])} for s in _lines(r, path, cache)]
    fresh_voices = cache.get("voices_at", 0) >= started
    t2 = time.perf_counter()

    def timing(review_s: float = 0.0) -> dict:
        """How long each step took. A step reused from an earlier run shows its original time, marked reused."""
        return {"transcribe_s": cache.get("transcribe_s") if not fresh_stt else round(t1 - t0, 1),
                "transcribe_reused": not fresh_stt, "voices_s": cache.get("voices_s"), "voices_reused": not fresh_voices,
                "lines_s": round(t2 - t1, 1), "review_s": round(review_s, 1), "total_s": round(time.perf_counter() - t0, 1),
                "stt": cache.get("stt")}

    fields = {"status": "done", "processed_at": time.time(), "attempts": r["attempts"] + 1, "error": None}
    if not segs:
        db.update("recordings", r["id"], **fields, is_csp_call=0, transcript_json=[], timing_json=timing(),
                  summary="No speech in the recording")
        integrations.recording_processed(r["id"])
        return
    t3 = time.perf_counter()
    review, model = review_call(segs) if llm_engine.provider() else ({}, None)
    review_s = time.perf_counter() - t3
    speakers = review.get("speakers") or ["unknown"] * len(segs)
    db.update("recordings", r["id"], **fields, timing_json=timing(review_s),
              transcript_json=[{**s, "speaker": sp} for s, sp in zip(segs, speakers)],
              is_csp_call=None if not review else int(bool(review.get("is_csp_call"))),
              language=review.get("language"), issue_segment=review.get("issue_segment"),
              intent=review.get("intent"), intent_confidence=review.get("intent_confidence"),
              slots_json=review.get("slots") or {}, sentiment=review.get("sentiment"),
              resolution=review.get("resolution"), rm_handling=review.get("rm_handling"),
              summary=review.get("summary"), model=model)
    _write_summary(r["id"])
    integrations.recording_processed(r["id"])


def _write_summary(rec_id: int):
    """The full summary (app/report.py). A failure here never fails the recording: the page offers a retry."""
    if not llm_engine.provider():
        return
    try:
        report.make_summary(rec_id)
    except Exception:
        log.exception("Summary failed for recording %d", rec_id)


_busy = threading.Lock()


def process_pending(limit: int | None = None, sleep=time.sleep, now: float | None = None) -> dict:
    """Transcribe and review every new recording that is due, oldest first. When a free-tier limit is hit the
    batch waits and retries a few times, then stops; whatever is left stays new for the next run."""
    if not _busy.acquire(blocking=False):
        return {"busy": True}
    try:
        if not stt.available():
            return {"processed": 0, "failed": 0, "left": _count_new(), "stopped": "GROQ_API_KEY is not set"}
        rows = db.q("SELECT * FROM recordings WHERE status='new' AND COALESCE(next_try_at, 0) <= ? ORDER BY uploaded_at"
                    + (f" LIMIT {int(limit)}" if limit else ""), (now or time.time(),))
        done = failed = 0
        stopped = None
        for r in rows:
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                try:
                    process_one(r)
                    done += 1
                except RateLimited as e:
                    if attempt < RATE_LIMIT_RETRIES:
                        log.warning("%s; waiting %.0fs before recording %d", e, min(e.wait, 120), r["id"])
                        sleep(min(e.wait, 120))
                        continue
                    stopped = str(e)
                except Exception as e:
                    log.exception("Recording %d failed", r["id"])
                    attempts = r["attempts"] + 1
                    gave_up = attempts >= MAX_ATTEMPTS
                    db.update("recordings", r["id"], attempts=attempts, error=str(e)[:500],
                              status="failed" if gave_up else "new",
                              next_try_at=None if gave_up else (now or time.time()) + RETRY_AFTER_S * attempts)
                    failed += gave_up
                break
            if stopped:
                break
        return {"processed": done, "failed": failed, "left": _count_new(), "stopped": stopped}
    finally:
        _busy.release()


def _count_new() -> int:
    return db.q("SELECT COUNT(*) AS n FROM recordings WHERE status='new'", one=True)["n"]


_arrived = threading.Event()


def wake():
    """A recording arrived (or the app is stopping): let the processing thread look now."""
    _arrived.set()


def due(now: datetime, last_day: str | None, daily_at: str | None = None) -> bool:
    """Daily mode: true once a day, at or after RECORDINGS_DAILY_AT."""
    hh, mm = (int(p) for p in (daily_at or DAILY_AT).split(":"))
    return now.strftime("%Y-%m-%d") != last_day and (now.hour, now.minute) >= (hh, mm)


def processing_loop(stop: threading.Event):
    """Background thread (started by app.main). By default every recording is processed within seconds of
    its upload; a check every 5 minutes also picks up retries. With RECORDINGS_DAILY_AT set, it runs once a day
    instead. If a free-tier limit stops a run, it tries again 30 minutes later."""
    last_day, next_try = None, 0.0
    while not stop.is_set():
        if not DAILY_AT:
            if time.time() >= next_try and _count_new():
                try:
                    res = process_pending()
                    if res.get("processed") or res.get("stopped"):
                        log.warning("Recordings run: %s", res)
                    next_try = time.time() + 1800 if res.get("stopped") else 0.0
                except Exception:
                    log.exception("Recordings run failed")
                    next_try = time.time() + 1800
            _arrived.wait(300)
            _arrived.clear()
            continue
        now = datetime.now(TZ)
        if due(now, last_day) and time.time() >= next_try:
            try:
                res = process_pending()
                log.warning("Daily recordings run: %s", res)
                if res.get("stopped") and res.get("left") and stt.available():
                    next_try = time.time() + 1800
                elif not res.get("busy"):
                    last_day = now.strftime("%Y-%m-%d")
            except Exception:
                log.exception("Daily recordings run failed")
                next_try = time.time() + 1800
        stop.wait(300)


# ---------------------------------------------------------------- reading results
LIST_COLS = """r.id, r.rm_id, m.name AS rm_name, r.csp_id, c.name AS csp_name, c.csp_code, r.other_number, r.started_at,
               r.duration_sec, r.direction, r.uploaded_at, r.status, r.error, r.attempts, r.is_csp_call, r.language,
               r.intent, r.sentiment, r.resolution, r.summary, r.orig_name,
               CASE WHEN r.token_id IS NULL THEN 'manual' ELSE 'phone' END AS source"""


def list_recordings(status: str | None = None, rm_id: int | None = None, limit: int = 200) -> list[dict]:
    where, args = [], []
    if status:
        where.append("r.status=?")
        args.append(status)
    if rm_id:
        where.append("r.rm_id=?")
        args.append(rm_id)
    sql = (f"SELECT {LIST_COLS} FROM recordings r LEFT JOIN rms m ON m.id=r.rm_id LEFT JOIN csps c ON c.id=r.csp_id"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY r.started_at DESC, r.id DESC LIMIT ?")
    return db.q(sql, (*args, max(1, min(limit, 1000))))


def get_recording(rec_id: int) -> dict | None:
    r = db.q(f"""SELECT {LIST_COLS}, r.transcript_json, r.slots_json, r.issue_segment, r.intent_confidence, r.rm_handling,
                 r.model, r.processed_at, r.next_try_at, r.timing_json, r.report_md, r.report_error, r.report_at,
                 r.report_s, r.report_model FROM recordings r LEFT JOIN rms m ON m.id=r.rm_id
                 LEFT JOIN csps c ON c.id=r.csp_id WHERE r.id=?""", (rec_id,), one=True)
    if r:
        r["timing"] = json.loads(r.pop("timing_json") or "null")
        r["transcript"] = json.loads(r.pop("transcript_json") or "[]")
        r["slots"] = json.loads(r.pop("slots_json") or "{}")
    return r


def summary_counts(rm_id: int | None = None) -> dict:
    rows = db.q("SELECT status, COUNT(*) AS n FROM recordings" + (" WHERE rm_id=?" if rm_id else "") + " GROUP BY status",
                (rm_id,) if rm_id else ())
    return {r["status"]: r["n"] for r in rows}


# ---------------------------------------------------------------- feeding the scorecard
def export_candidates(out_csv: str | None = None, audio_dir: str | None = None, max_clip_s: float = 25) -> int:
    """Cut the CSP's opening complaint from every finished call into evals/audio/ and add a draft row to
    evals/candidates.csv. reference_text is left empty on purpose: someone listens and writes what was
    really said, then moves the row to cases.csv (Whisper's own text would score itself as perfect)."""
    out_csv = out_csv or os.path.join(ROOT, "evals", "candidates.csv")
    audio_dir = audio_dir or os.path.join(ROOT, "evals", "audio")
    with open(os.path.join(ROOT, "evals", "cases.csv"), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols, have = reader.fieldnames, {row["id"] for row in reader}
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            have |= {row["id"] for row in csv.DictReader(f)}
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Cutting clips needs ffmpeg")
    os.makedirs(audio_dir, exist_ok=True)
    new_rows = []
    for r in db.q("""SELECT * FROM recordings WHERE status='done' AND is_csp_call=1 AND issue_segment >= 0
                     ORDER BY id"""):
        case_id = f"rec{r['id']}"
        segs = json.loads(r["transcript_json"] or "[]")
        i = r["issue_segment"]
        if case_id in have or i >= len(segs):
            continue
        j, end = i + 1, segs[i]["end"]
        while j < len(segs) and segs[j].get("speaker") == "csp" and segs[j]["end"] - segs[i]["start"] <= max_clip_s:
            end, j = segs[j]["end"], j + 1
        clip = f"{case_id}.ogg"
        cut = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                              "-ss", f"{max(segs[i]['start'] - 0.2, 0):.2f}", "-to", f"{end + 0.3:.2f}",
                              "-i", os.path.join(storage_dir(), r["file_path"]), "-ac", "1", "-ar", "16000",
                              "-c:a", "libopus", "-b:a", "32k", os.path.join(audio_dir, clip)],
                             capture_output=True, timeout=120)
        if cut.returncode != 0:
            log.warning("Couldn't cut a clip from recording %d: %s", r["id"], cut.stderr[:200])
            continue
        heard = " ".join(s["text"] for s in segs[i:j])
        row = dict.fromkeys(cols, "")
        row.update(id=case_id, lang="en" if r["language"] == "en" else "hi", audio=clip,
                   expected_intent=r["intent"] if r["intent"] in CATEGORIES else "",
                   notes=f"DRAFT from recorded call {r['id']}: listen and fill reference_text, expected_action and "
                         f"details. Whisper heard: {heard} | AI review of the whole call: {r['summary']} "
                         f"{r['slots_json'] or ''}")
        new_rows.append(row)
    if new_rows:
        fresh = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if fresh:
                w.writeheader()
            w.writerows(new_rows)
    return len(new_rows)


# ---------------------------------------------------------------- command line
def _cli():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    p = argparse.ArgumentParser(prog="python -m app.recordings", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("token", help="make an upload token for one RM's phone")
    t.add_argument("--rm", type=int, help="the RM's id; leave out for a shared token (each RM picks their name in the app)")
    t.add_argument("--label", default="")
    sub.add_parser("tokens", help="list upload tokens")
    rv = sub.add_parser("revoke", help="revoke an upload token")
    rv.add_argument("id", type=int)
    pr = sub.add_parser("process", help="process every new recording now")
    pr.add_argument("--limit", type=int)
    sub.add_parser("list", help="list recordings")
    sub.add_parser("export", help="draft scorecard cases -> evals/candidates.csv")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .seed import seed_if_demo
    seed_if_demo()                           # as when the server starts
    if a.cmd == "token":
        tok = create_token(a.rm or None, a.label, "cli")
        who = tok["rm_name"] or "any RM (they choose their name in the app)"
        print(f"Upload token for {who} ({tok['label'] or 'no label'}). Shown only once:\n\n  {tok['token']}\n")
    elif a.cmd == "tokens":
        for t in list_tokens():
            state = "revoked" if t["revoked_at"] else "active"
            print(f"{t['id']:>3}  {state:7}  {t['rm_name']:<20} {t['label'] or '-':<24} uploads {t['uploads']}")
    elif a.cmd == "revoke":
        print("Revoked" if revoke_token(a.id) else "No active token with that id")
    elif a.cmd == "process":
        print(process_pending(a.limit))
    elif a.cmd == "list":
        for r in list_recordings():
            when = datetime.fromtimestamp(r["started_at"] or r["uploaded_at"], TZ).strftime("%d %b %H:%M")
            print(f"{r['id']:>4}  {when}  {r['rm_name'] or '-':<16} {r['status']:<6} {r['intent'] or '-':<15} "
                  f"{(r['summary'] or r['error'] or '')[:70]}")
    elif a.cmd == "export":
        print(f"Added {export_candidates()} draft case(s) to evals/candidates.csv")


if __name__ == "__main__":
    _cli()
