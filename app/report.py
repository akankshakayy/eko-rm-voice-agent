"""A full written summary of a recording, and the files people download: transcript, summary, Word document.

The call review in recordings.py sees only the start of a long transcript (the LLM's free tier takes ~3k tokens
at a time). This summary reads all of it: the transcript is cut into parts, each part is turned into dated notes,
then the notes become one summary. The models are told to keep only what was said, copy numbers and names exactly,
and mark anything unclear, so the summary can be checked against the transcript line by line (every point carries
its [mm:ss] time).
"""
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

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


# ---------------------------------------------------------------- Word (.docx)
# A real .docx (a zip of WordprocessingML), built with the standard library. HTML saved as .doc opened only in
# desktop Word; LibreOffice, Google Docs and phone apps showed it empty. Hindi uses the complex-script font.
_FONTS = '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Nirmala UI"/>'
_STYLES = {"Title": 36, "Heading1": 28, "Heading2": 24, "Heading3": 22}


def _run(text: str, bold: bool = False, color: str | None = None) -> str:
    props = _FONTS + ("<w:b/><w:bCs/>" if bold else "") + (f'<w:color w:val="{color}"/>' if color else "")
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r>'


def _runs(text: str) -> str:
    """**bold** inside a line becomes bold runs; everything else is plain text."""
    parts = re.split(r"\*\*(.+?)\*\*", text)
    return "".join(_run(p, bold=i % 2 == 1) for i, p in enumerate(parts) if p)


def _para(runs: str, style: str | None = None, bullet: bool = False) -> str:
    props = (f'<w:pStyle w:val="{style}"/>' if style else "") + \
        ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>' if bullet else "") + \
        '<w:spacing w:after="60"/>'
    return f"<w:p><w:pPr>{props}</w:pPr>{runs}</w:p>"


def _md_paras(md: str) -> list[str]:
    """The small Markdown subset the summary uses: # headings, - bullets, **bold**."""
    out = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            out.append(_para(_runs(line.lstrip("#").strip()), f"Heading{level}"))
        elif line.startswith(("- ", "* ")):
            out.append(_para(_runs(line[2:]), bullet=True))
        else:
            out.append(_para(_runs(line)))
    return out


def docx_file(rec: dict) -> bytes:
    """Summary first, then the full transcript, as a Word document."""
    csp = bool(rec.get("is_csp_call"))
    body = [_para(_run("Call recording"), "Title")]
    body += [_para(_run(h)) for h in header(rec)]
    body.append(_para(_run("Summary"), "Heading1"))
    body += _md_paras(rec.get("report_md") or "No summary yet.")
    body.append(_para(_run("Transcript"), "Heading1"))
    for l in rec.get("transcript") or []:
        body.append(_para(_run(f"[{clock(l['start'])}] ", color="666666") + _run(f"{speaker(l, csp)}: ", bold=True)
                          + _run(l["text"])))
    W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W}><w:body>{"".join(body)}'
                '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
                'w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>')
    styles = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {W}><w:docDefaults><w:rPrDefault>'
              f'<w:rPr>{_FONTS}<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
              + "".join(f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{sid.replace("Heading", "heading ").lower() if sid != "Title" else "Title"}"/>'
                        f'<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/>'
                        f'{"<w:outlineLvl w:val=" + chr(34) + str(int(sid[-1]) - 1) + chr(34) + "/>" if sid != "Title" else ""}</w:pPr>'
                        f'<w:rPr><w:b/><w:bCs/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr></w:style>'
                        for sid, sz in _STYLES.items()) + "</w:styles>")
    numbering = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering {W}><w:abstractNum w:abstractNumId="0">'
                 '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/>'
                 '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>'
                 '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>')
    R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
            '</Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/_rels/document.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{R}/styles" Target="styles.xml"/>'
            f'<Relationship Id="rId2" Type="{R}/numbering" Target="numbering.xml"/></Relationships>',
        "word/document.xml": document, "word/styles.xml": styles, "word/numbering.xml": numbering}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()
