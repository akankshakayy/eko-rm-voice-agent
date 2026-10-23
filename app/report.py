"""A full written summary of a recording, and the files people download: transcript, summary, Word document.

The call review in recordings.py sees only the start of a long transcript (the LLM's free tier takes ~3k tokens
at a time). This summary reads all of it: the transcript is cut into parts, each part is turned into dated notes,
then the notes become one summary. The models are told to keep only what was said, copy numbers and names exactly,
and mark anything unclear, so the summary can be checked against the transcript line by line (every point carries
its [mm:ss] time).
"""
import html
import json
import os
import re
import time
from datetime import datetime

from . import db, llm_engine

CHUNK_CHARS = 6000          # one part of the transcript per LLM call (fits a Groq free-tier minute)
# gpt-oss takes 8k tokens a minute on Groq's free tier; qwen only 1k output tokens a minute, too few for notes
SUMMARY_MODELS = os.environ.get("SUMMARY_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b")
# These models think before answering; turned down, the thinking doesn't eat the answer's token allowance
EXTRA = {"openai/gpt-oss": {"reasoning_effort": "low"}, "qwen/": {"reasoning_effort": "none"}}
SPEAKER_WORDS = {"rm": "RM", "csp": "CSP", "other": "Other person"}


class RateLimited(Exception):
    def __init__(self, message: str, wait: float = 60):
        super().__init__(message)
        self.wait = wait


def clock(s: float) -> str:
    s = int(s or 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def speaker(line: dict, csp_call: bool) -> str:
    """Speaker 1, 2… from the voice; on RM–CSP calls also the role, e.g. "Speaker 1 (RM)"."""
    v = line.get("voice")
    name = f"Speaker {ord(v) - 64}" if v else SPEAKER_WORDS.get(line.get("speaker"), "Speaker")
    role = SPEAKER_WORDS.get(line.get("speaker")) if csp_call and v else None
    return f"{name} ({role})" if role else name


def transcript_text(rec: dict) -> str:
    csp = bool(rec.get("is_csp_call"))
    return "\n".join(f"[{clock(l['start'])}] {speaker(l, csp)}: {l['text']}" for l in rec.get("transcript") or [])


# ---------------------------------------------------------------- the summary
PART_PROMPT = """You write one part of a detailed summary of a recorded conversation (Hindi, English or Hinglish).
Each line is "[mm:ss] Speaker N: text". It comes from speech-to-text, so some words are misheard. Where a common
word was clearly misheard (e.g. "आई से" for "AI से", "क्लोड" for "Claude"), write the intended word. Never guess
numbers, amounts, dates, IDs or names: copy them exactly, or mark them unclear.

The reader did not hear the recording and must learn everything that matters in this part. Reply in English with
exactly these two blocks:

TOPICS
One ### heading per topic, in order, with its time range, e.g. "### Testing after every change [03:11–03:43]"
(no numbering).
Under each, bullets explaining what was said and by whom (Speaker 1, Speaker 2, ...), with the reasons, examples,
steps and details given. Each bullet starts with its [mm:ss] time. Leave out filler (greetings, "OK", "है ना?",
single words) unless it matters.

EXTRACTS
Bullets, each starting with one label:
"Decision:" something agreed.
"Action:" something a person agreed or was asked to do after or outside this conversation (who, what, by when if
said). Not the steps someone shows or clicks through on screen during the call.
"Number:" an exact value and what it is. "Name:" a person, product, tool or website and what it is.
"Open:" a question that was asked and left unanswered, or an important passage too unclear to follow. Skip minor
unclear words. Write "None" if there are none.

Use only what was said. Do not invent, generalise or soften anything."""

FINAL_PROMPT = """You finish a detailed summary of a recorded conversation. You get the list of topics (with times)
and every decision, action, number, name and open point taken from all of it. Use only that. Do not invent anything.
Keep [mm:ss] times and speaker names (Speaker 1, Speaker 2, ...) exactly. Write in clear, plain English, with exactly
these Markdown sections, leaving out a section only if there is nothing for it:

## In short
One paragraph of 5 to 8 sentences: who talked about what, why, and how it ended.

## Decisions
## Action items
Bullets: who will do what afterwards, and by when if said. Merge repeats; leave out on-screen demo steps.

## Numbers, names and IDs mentioned
Bullets with the exact value and what it refers to. Merge repeats.

## Open questions and unclear parts
At most 8 bullets: real unanswered questions first, then only the unclear passages that matter."""


def _chat(system: str, user: str, max_tokens: int = 2500) -> tuple[str, str, bool]:
    """One plain completion on the configured LLM -> (text, model, cut off?). When every model in the chain is
    rate-limited, RateLimited."""
    p = llm_engine.provider()
    if p is None:
        raise ValueError("No LLM key is set (GROQ_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY)")
    if p == "claude":
        import anthropic
        try:
            resp = anthropic.Anthropic().messages.create(model=llm_engine.MODEL, max_tokens=max_tokens, system=system,
                                                         messages=[{"role": "user", "content": user}])
        except anthropic.RateLimitError as e:
            raise RateLimited("Claude rate limit") from e
        return ("".join(b.text for b in resp.content if b.type == "text").strip(), llm_engine.MODEL,
                resp.stop_reason == "max_tokens")
    from openai import APIStatusError, OpenAI, RateLimitError
    cfg = llm_engine.PROVIDERS[p]
    client = OpenAI(api_key=os.environ.get(cfg["key_env"]), base_url=cfg["base_url"], timeout=180, max_retries=0)
    chain = SUMMARY_MODELS if p == "groq" else cfg["model"]
    models = [m.strip() for m in chain.split(",") if m.strip()]
    waits = []
    for i, model in enumerate(models):
        last = i == len(models) - 1
        extra = next((v for k, v in EXTRA.items() if model.startswith(k)), {})
        try:
            resp = client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=0, messages=[
                {"role": "system", "content": system}, {"role": "user", "content": user}], extra_body=extra or None)
        except RateLimitError as e:           # this model's minute is used up: try the next, else wait
            m = re.search(r"try again in ([\d.]+)s", str(e))
            waits.append(float(m.group(1)) + 1 if m else 30)
            if last:
                raise RateLimited(f"{cfg['label']} rate limit", min(waits)) from e
            continue
        except APIStatusError as e:              # a part too long for this model: the next one may take it
            if last or e.status_code not in (400, 413):
                raise
            continue
        text = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.S).strip()
        if text:
            return text, model, resp.choices[0].finish_reason == "length"
    raise ValueError("No model returned a summary")


def _parts(text: str) -> list[str]:
    parts, cur = [], []
    for line in text.splitlines():
        if cur and sum(len(x) + 1 for x in cur) + len(line) > CHUNK_CHARS:
            parts.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(cur))
    return parts


def _blocks(text: str) -> tuple[str, str]:
    """Split a part's answer into its TOPICS and EXTRACTS blocks."""
    m = re.split(r"(?im)^[#*\s]*EXTRACTS[*:\s]*$", text, maxsplit=1)
    topics = re.sub(r"(?im)^[#*\s]*TOPICS[*:\s]*$", "", m[0]).strip()
    topics = re.sub(r"(?m)^#{1,2}(?!#)\s*", "### ", topics)              # topic headings are always ###
    topics = re.sub(r"(?m)^### (?:\d+[.)]\s*)", "### ", topics)        # and never numbered
    return topics, (m[1].strip() if len(m) > 1 else "")


def summarize(rec: dict, sleep=time.sleep, retries: int = 8) -> tuple[str, str]:
    """The whole transcript -> one detailed Markdown summary. Returns (markdown, model).
    Each part of the transcript writes its own topic sections, which go into the summary unchanged, so nothing is
    lost however long the call is; one small final request writes the overview, decisions, actions and names.
    A part whose answer gets cut off is split in two and redone. Free-tier limits are waited out."""
    def call(system, user, max_tokens):
        for attempt in range(retries + 1):
            try:
                out = _chat(system, user, max_tokens)
                return out[0], out[1], (out[2] if len(out) > 2 else False)
            except RateLimited as e:
                if attempt == retries:
                    raise
                sleep(min(e.wait, 90))

    def part_summary(part, label, depth=0):
        text, model, cut = call(PART_PROMPT, f"{label} of the transcript:\n\n{part}", 2600)
        lines = part.splitlines()
        if cut and depth < 2 and len(lines) > 1:
            half = len(lines) // 2
            a = part_summary("\n".join(lines[:half]), label + " (first half)", depth + 1)
            b = part_summary("\n".join(lines[half:]), label + " (second half)", depth + 1)
            return a[0] + "\n\n" + b[0], a[1] + "\n" + b[1], b[2]
        return (*_blocks(text), model)

    text = transcript_text(rec)
    if not text.strip():
        return "## In short\nNo speech was found in this recording.", ""
    parts = _parts(text)
    topics, extracts, model = [], [], ""
    for i, part in enumerate(parts, 1):
        t, e, model = part_summary(part, f"Part {i} of {len(parts)}")
        topics.append(t)
        if e and e.strip().lower() != "none":
            extracts.append(e)
    headings = "\n".join(l for t in topics for l in t.splitlines() if l.startswith("###"))
    final, model, _ = call(FINAL_PROMPT, f"Topics, in order:\n{headings}\n\nTaken from all parts:\n" + "\n".join(extracts), 2000)
    m = re.search(r"(?ms)^##\s*In short.*?(?=^##\s)", final + "\n## ")
    in_short = (m.group(0).strip() if m else "## In short\n" + final.split("\n## ")[0].strip())
    rest = final[m.end():].strip() if m and m.end() <= len(final) else ""
    md = f"{in_short}\n\n## Topic by topic\n\n" + "\n\n".join(t for t in topics if t) + (f"\n\n{rest}" if rest else "")
    return md, model


def make_summary(rec_id: int) -> dict:
    """Write the summary for one recording and store it (errors are stored too, for the page to show)."""
    from .recordings import get_recording
    rec = get_recording(rec_id)
    t0 = time.perf_counter()
    try:
        md, model = summarize(rec)
        db.update("recordings", rec_id, report_md=md, report_error=None, report_at=time.time(),
                  report_s=round(time.perf_counter() - t0, 1), report_model=model)
    except Exception as e:
        db.update("recordings", rec_id, report_error=f"{type(e).__name__}: {e}"[:400])
        raise
    return {"ok": True}


# ---------------------------------------------------------------- downloads
def filename(rec: dict, ext: str) -> str:
    when = datetime.fromtimestamp(rec.get("started_at") or rec.get("uploaded_at") or time.time()).strftime("%Y-%m-%d %H%M")
    who = rec.get("csp_name") or rec.get("other_number") or os.path.splitext(rec.get("orig_name") or "recording")[0]
    return re.sub(r"[^\w\- ]+", "", f"{when} {who}")[:80].strip() + f".{ext}"


def header(rec: dict) -> list[str]:
    when = datetime.fromtimestamp(rec.get("started_at") or rec.get("uploaded_at") or 0).strftime("%d %b %Y, %H:%M")
    rows = [f"Recording: {rec.get('orig_name') or rec['id']}", f"Date: {when}",
            f"Length: {clock(rec.get('duration_sec') or 0)}"]
    if rec.get("rm_name"):
        rows.append(f"RM: {rec['rm_name']}")
    if (rec.get("timing") or {}).get("stt"):
        rows.append(f"Transcribed by: {rec['timing']['stt']}")
    return rows


def transcript_file(rec: dict) -> str:
    return "\n".join(header(rec)) + "\n\n" + transcript_text(rec) + "\n"


def summary_file(rec: dict) -> str:
    return "# Summary\n\n" + "\n".join(f"- {h}" for h in header(rec)) + "\n\n" + (rec.get("report_md") or "") + "\n"


def _md_html(md: str) -> str:
    """The small Markdown subset the summary uses (headings, bullets, bold) -> HTML, everything else escaped."""
    out, in_list = [], False
    for raw in md.splitlines():
        line = html.escape(raw.strip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        if line.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 4)      # ## -> h2, ### -> h3
            out.append(f"<h{level}>{line.lstrip('#').strip()}</h{level}>")
        elif line:
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def word_file(rec: dict) -> str:
    """A Word document (Word opens HTML saved as .doc): summary first, then the full transcript."""
    csp = bool(rec.get("is_csp_call"))
    lines = "\n".join(f"<p><span style='color:#666'>[{clock(l['start'])}]</span> <b>{html.escape(speaker(l, csp))}:</b> "
                      f"{html.escape(l['text'])}</p>" for l in rec.get("transcript") or [])
    return f"""<html><head><meta charset="utf-8"><title>{html.escape(filename(rec, 'doc'))}</title>
<style>body {{ font-family: Calibri, Arial, sans-serif; font-size: 11pt; }} h1 {{ font-size: 18pt; }}
h2, h3 {{ font-size: 13pt; margin-top: 14pt; }} p {{ margin: 3pt 0; }}</style></head><body>
<h1>Call recording</h1><p>{"<br>".join(html.escape(h) for h in header(rec))}</p>
<h1>Summary</h1>{_md_html(rec.get("report_md") or "No summary yet.")}
<h1>Transcript</h1>{lines}</body></html>"""
