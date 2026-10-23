"""Sarvam AI speech-to-text for recorded calls: transcription and speaker separation (diarization) in one step.

Sarvam's models are built for Indian languages and Hinglish, and its servers are in India. Speaker separation is
only offered through its batch API, which suits recordings (they aren't live): create a job, upload the file, start
it, wait, then download the result. Each entry of the result is one speaker's stretch of speech with its times.

    SARVAM_API_KEY=...            # from dashboard.sarvam.ai; without it recordings use Whisper + the local voice model
    SARVAM_STT_MODEL=saaras:v4    # Sarvam's current speech model
    SARVAM_STT_MODE=codemix       # Hindi-English mixed speech, English words kept in English letters
"""
import os
import shutil
import subprocess
import tempfile
import time

API = "https://api.sarvam.ai/speech-to-text/job/v1"
MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v4")
MODE = os.environ.get("SARVAM_STT_MODE", "codemix")
LANGUAGE = os.environ.get("SARVAM_STT_LANGUAGE", "hi-IN")
POLL_S, MAX_WAIT_S = 5, 45 * 60


class SarvamError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status_code = status        # 429 is treated as a rate limit by recordings.py


def available() -> bool:
    return bool(os.environ.get("SARVAM_API_KEY"))


def settings() -> str:
    """What a saved result depends on; a change here means the file is sent again."""
    return f"sarvam|{MODEL}|{MODE}|{LANGUAGE}|{os.environ.get('RECORDINGS_SPEAKERS', '0')}"


def _client():
    import httpx
    return httpx.Client(timeout=120, headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]})


def _check(r, what: str):
    if r.status_code >= 400:
        raise SarvamError(f"Sarvam {what} failed ({r.status_code}): {r.text[:300]}", r.status_code)
    return r.json() if r.content else {}


def _as_mp3(path: str) -> str:
    """Phone and WhatsApp formats vary; a 16 kHz mono MP3 is small to upload and always accepted."""
    if not shutil.which("ffmpeg"):
        return path
    out = os.path.join(tempfile.mkdtemp(prefix="sarvam-"), "call.mp3")
    done = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000",
                           "-b:a", "48k", out], capture_output=True, timeout=600)
    if done.returncode != 0:
        raise ValueError(f"Couldn't read this audio file: {done.stderr.decode(errors='replace')[:200]}")
    return out


def transcribe(path: str, sleep=time.sleep) -> list[dict]:
    """The whole recording as lines, one speaker each: [{"start", "end", "text", "voice": "A"}]."""
    mp3 = _as_mp3(path)
    name = "call.mp3" if mp3 != path else os.path.basename(path)
    params = {"model": MODEL, "mode": MODE, "language_code": LANGUAGE, "with_diarization": True}
    speakers = int(os.environ.get("RECORDINGS_SPEAKERS", "0") or 0)
    if speakers:
        params["num_speakers"] = speakers
    with _client() as c:
        job = _check(c.post(API, json={"job_parameters": params}), "job creation")["job_id"]
        urls = _check(c.post(f"{API}/upload-files", json={"job_id": job, "files": [name]}), "upload request")
        url = (urls.get("upload_urls") or {}).get(name, {}).get("file_url")
        if not url:
            raise SarvamError(f"Sarvam gave no upload address: {str(urls)[:200]}")
        with open(mp3, "rb") as f:
            _check(c.put(url, content=f.read(), headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "audio/mpeg"},
                         timeout=600), "upload")
        _check(c.post(f"{API}/{job}/start"), "job start")
        waited = 0
        while True:
            status = _check(c.get(f"{API}/{job}/status"), "status check")
            state = status.get("job_state")
            if state == "Completed":
                break
            if state == "Failed" or waited > MAX_WAIT_S:
                details = "; ".join(str(d.get("error_message") or "") for d in status.get("job_details") or [])
                raise SarvamError(f"Sarvam job {state or 'timed out'}: {status.get('error_message') or details}")
            sleep(POLL_S)
            waited += POLL_S
        files = [o["file_name"] for d in status.get("job_details") or [] for o in d.get("outputs") or []]
        if not files:
            raise SarvamError("Sarvam finished but returned no transcript")
        links = _check(c.post(f"{API}/download-files", json={"job_id": job, "files": files}), "download request")
        file_url = next(iter((links.get("download_urls") or {}).values()), {}).get("file_url")
        if not file_url:
            raise SarvamError(f"Sarvam gave no download address: {str(links)[:200]}")
        import httpx
        result = _check(httpx.get(file_url, timeout=120), "download")
    if mp3 != path:
        shutil.rmtree(os.path.dirname(mp3), ignore_errors=True)
    return lines_from(result)


def lines_from(result: dict) -> list[dict]:
    """Sarvam's output -> transcript lines. Speakers "0", "1"… become voices A, B… in order of first appearance."""
    entries = ((result.get("diarized_transcript") or {}).get("entries")) or []
    letters, lines = {}, []
    for e in entries:
        text = (e.get("transcript") or "").strip()
        if not text:
            continue
        v = letters.setdefault(str(e.get("speaker_id")), chr(ord("A") + len(letters)))
        lines.append({"start": round(float(e.get("start_time_seconds") or 0), 2),
                      "end": round(float(e.get("end_time_seconds") or 0), 2), "text": text, "voice": v})
    if not lines and (result.get("transcript") or "").strip():        # no speaker split came back: one block
        lines.append({"start": 0.0, "end": 0.0, "text": result["transcript"].strip(), "voice": None})
    return lines
