"""Who spoke when, from the voices themselves (speaker diarization), on this machine's own CPU.

Whisper cuts a call at pauses, not where the speaker changes, so on its own one line can hold the RM's question
and the CSP's answer. This module finds the voice turns (voice A, voice B) with two small local models from
sherpa-onnx: pyannote segmentation (when someone is speaking) and a speaker-embedding model (whose voice it is).
About 35 MB in all, no account or GPU needed, and no audio leaves the machine. assign_words() then gives every
word Whisper heard to the voice speaking at that moment, so a line never mixes two people. The LLM only decides
which voice is the RM and which the CSP.

    python -m app.diarize download          # fetch the models into data/models/ (once per machine)
    python -m app.diarize some_call.m4a     # print the voice turns of one file

Without the models or the sherpa-onnx package, available() is False and recordings fall back to sentence splitting.
"""
import bisect
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request

from . import db

log = logging.getLogger("eko.diarize")
# How many voices to look for: 0 = let the model decide (a shop customer or a third person can join a call)
SPEAKERS = int(os.environ.get("RECORDINGS_SPEAKERS", "0"))
# With SPEAKERS=0: how different two voices must sound to count as two people (lower = splits more readily)
THRESHOLD = float(os.environ.get("RECORDINGS_SPEAKER_THRESHOLD", "0.5"))
RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
SEG_URL = f"{RELEASES}/speaker-segmentation-models/{SEG_DIR}.tar.bz2"
EMB_FILE = "wespeaker_en_voxceleb_resnet34_LM.onnx"           # trained on VoxCeleb: many languages and accents
EMB_URL = f"{RELEASES}/speaker-recongition-models/{EMB_FILE}"  # sic: the release tag is spelled this way
SAMPLE_RATE = 16000
NEW_LINE_GAP_S = 1.2       # a pause this long starts a new line even when the same voice carries on
_lock = threading.Lock()
_engine = None


def models_dir() -> str:
    return os.environ.get("RECORDINGS_MODELS_DIR") or os.path.join(os.path.dirname(os.path.abspath(db.DB_PATH)), "models")


def _paths() -> tuple[str, str]:
    d = models_dir()
    return os.path.join(d, SEG_DIR, "model.onnx"), os.path.join(d, EMB_FILE)


def available() -> bool:
    try:
        import numpy  # noqa: F401
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return all(os.path.exists(p) for p in _paths()) and bool(shutil.which("ffmpeg"))


def download():
    """Fetch the two models (about 35 MB) into models_dir()."""
    d = models_dir()
    os.makedirs(d, exist_ok=True)
    seg, emb = _paths()
    if not os.path.exists(seg):
        tmp = os.path.join(d, "seg.tar.bz2")
        urllib.request.urlretrieve(SEG_URL, tmp)
        with tarfile.open(tmp) as t:
            t.extractall(d, filter="data")
        os.remove(tmp)
    if not os.path.exists(emb):
        urllib.request.urlretrieve(EMB_URL, emb + ".part")
        os.replace(emb + ".part", emb)


def _get_engine():
    global _engine
    if _engine is None:
        import sherpa_onnx
        seg, emb = _paths()
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg), num_threads=2),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=2),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=SPEAKERS or -1, threshold=THRESHOLD),
            min_duration_on=0.2, min_duration_off=0.3)
        if not config.validate():
            raise RuntimeError("Speaker diarization models are invalid")
        _engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _engine


def _load(path: str):
    """Any audio file -> 16 kHz mono float32 samples, via ffmpeg (reads the file itself, so phone .m4a works)."""
    import numpy as np
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
                          "-f", "f32le", "pipe:1"], capture_output=True, timeout=600)
    if out.returncode != 0:
        raise ValueError(f"Couldn't read this audio file: {out.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(out.stdout, dtype=np.float32)


def settings() -> str:
    """What a saved voice result depends on; a change here means the voices are separated again."""
    return f"{EMB_FILE}|{SPEAKERS}|{THRESHOLD}"


def diarize(path: str) -> list[dict]:
    """Voice turns, in time order: [{"start": 0.0, "end": 3.4, "voice": "A"}, ...]."""
    samples = _load(path)
    if not len(samples):
        return []
    with _lock:                                  # one engine, used by one recording at a time
        result = _get_engine().process(samples).sort_by_start_time()
    letters, turns = {}, []
    for r in result:                             # name voices A, B... in order of first appearance
        v = letters.setdefault(r.speaker, chr(ord("A") + len(letters)))
        turns.append({"start": round(r.start, 2), "end": round(r.end, 2), "voice": v})
    return turns


def voice_at(turns: list[dict], t: float) -> str | None:
    """The voice speaking at time t; between turns, the nearest one."""
    if not turns:
        return None
    starts = [x["start"] for x in turns]
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and turns[i]["start"] <= t <= turns[i]["end"]:
        return turns[i]["voice"]
    near = [x for x in (turns[i] if i >= 0 else None, turns[i + 1] if i + 1 < len(turns) else None) if x]
    return min(near, key=lambda x: min(abs(t - x["start"]), abs(t - x["end"])))["voice"]


_PUNCT = re.compile(r"\W+")      # strips punctuation, the Hindi danda (।) included, the same way on both sides


def punctuate(words: list[dict], segments: list[dict]) -> list[dict]:
    """Whisper's word list has no punctuation; its segment text does. Where a segment's words line up one-for-one
    with its text, take the punctuated spelling from the text, so lines still end in । ? and ,"""
    out, wi = [], 0
    for s in segments:
        mine = []
        while wi < len(words) and words[wi]["start"] < s["end"] - 0.01:
            mine.append(words[wi])
            wi += 1
        tokens = s["text"].split()
        if len(tokens) == len(mine) and all(_PUNCT.sub("", t).lower() == _PUNCT.sub("", w["word"]).lower()
                                            for t, w in zip(tokens, mine)):
            mine = [{**w, "word": t} for w, t in zip(mine, tokens)]
        out.extend(mine)
    out.extend(words[wi:])
    return out


SENTENCE_END = ("।", "॥", "?", "!", ".")
FRAGMENT_WORDS, FRAGMENT_S = 2, 3.0    # runs this short are checked: a whole sentence, or the edge of someone else's?
PAUSE_S = 0.5                          # a pause this long between words counts as a sentence break
SNAP_WORDS = 4                         # a mid-sentence voice change moves to a sentence break this close, if any


def _voice_of(turns: list[dict], start: float, end: float) -> str | None:
    """The voice heard most during [start, end]; if none, the nearest turn's."""
    heard = {}
    for t in turns:
        if t["start"] < end and t["end"] > start:
            heard[t["voice"]] = heard.get(t["voice"], 0) + min(end, t["end"]) - max(start, t["start"])
    return max(heard, key=heard.get) if heard else voice_at(turns, (start + end) / 2)


def _runs(voices: list) -> list[list]:
    runs = []
    for i, v in enumerate(voices):
        if runs and runs[-1][2] == v:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, v])
    return runs


def _smooth(words: list[dict], voices: list) -> list:
    """Voice turns are a fraction of a second off at their edges, so the end of one person's sentence can land on
    the other person ("…बहुत बड़ा super" | "site है।"). Two passes fix that:
    1. a run of one or two words that only continues or starts someone else's sentence joins that speaker; a whole
       short sentence ("Yes." "हाँ जी।") keeps its own voice, since short replies are real;
    2. a voice change in the middle of a sentence moves to the nearest sentence break (। ? or a pause) nearby.
    """
    n = len(words)

    def brk(i):            # is there a sentence break right after word i?
        return i >= n - 1 or words[i]["word"].endswith(SENTENCE_END) or words[i + 1]["start"] - words[i]["end"] >= PAUSE_S

    runs = _runs(voices)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, (a, b, v) in enumerate(runs):
            if b - a > FRAGMENT_WORDS or words[b - 1]["end"] - words[a]["start"] > FRAGMENT_S:
                continue
            starts_sentence = a == 0 or brk(a - 1)
            if starts_sentence and brk(b - 1):
                continue                                   # a complete short sentence: someone really said it
            target = k - 1 if (not starts_sentence and k > 0) or k + 1 == len(runs) else k + 1
            if runs[target][2] != v:
                runs[k][2] = runs[target][2]
                changed = True
        runs = _runs([v for a, b, v in runs for _ in range(a, b)])
    for k in range(len(runs) - 1):                         # pass 2: move mid-sentence changes to a break
        a, b = runs[k][0], runs[k][1]                      # the change is between word b-1 and word b
        if brk(b - 1):
            continue
        near = [j for j in range(max(a + 1, b - SNAP_WORDS), min(runs[k + 1][1] - 1, b + SNAP_WORDS) + 1)
                if brk(j - 1)]
        if near:
            j = min(near, key=lambda j: abs(j - b))
            runs[k][1] = runs[k + 1][0] = j
    out = []
    for a, b, v in runs:
        out.extend([v] * (b - a))
    return out


def assign_words(words: list[dict], turns: list[dict]) -> list[dict]:
    """Group Whisper's timed words into lines by voice: a new line wherever the voice changes or after a long pause.
    words: [{"word", "start", "end"}]. Returns [{"start", "end", "text", "voice"}]."""
    words = [{**w, "word": (w.get("word") or "").strip()} for w in words if (w.get("word") or "").strip()]
    voices = _smooth(words, [_voice_of(turns, w["start"], w["end"]) for w in words]) if words else []
    lines = []
    for w, v in zip(words, voices):
        last = lines[-1] if lines else None
        if last and last["voice"] == v and w["start"] - last["end"] < NEW_LINE_GAP_S:
            last["text"] += " " + w["word"]
            last["end"] = round(w["end"], 2)
        else:
            lines.append({"start": round(w["start"], 2), "end": round(w["end"], 2), "text": w["word"], "voice": v})
    return lines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1:] == ["download"]:
        download()
        print("Models ready in", models_dir())
    else:
        for x in diarize(sys.argv[1]):
            print(f"{x['start']:7.2f} {x['end']:7.2f}  voice {x['voice']}")
