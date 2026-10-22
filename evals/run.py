"""Scorecard: how well does the agent hear and understand real CSP calls, and how fast?

    python -m evals.run                     # every case in evals/cases.csv
    python -m evals.run --synthesize        # also make audio (Edge voice) for cases that have none
    python -m evals.run --engine rules      # score the offline rules engine instead of the LLM
    python -m evals.run --only dmt_english,angry

For each case (one row of cases.csv, usually the caller's opening sentence):
  1. audio -> Whisper; scored against reference_text (WER/CER) and timed
  2. that transcript -> the agent's first turn; intent, details and action scored, LLM timed
  3. reference_text -> the agent too, so the gap between 2 and 3 is damage done by mishearing alone
  4. the agent's reply -> Edge voice, timed, for an end-to-end time per turn
Runs in a throwaway database, never the real one. Results go to evals/results/<timestamp>/.
"""
import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "evals")
os.environ["EKO_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="eko-eval-"), "eval.db")  # before app imports

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from app import agent, db, stt, tts  # noqa: E402
from app.seed import seed  # noqa: E402

CALLER = "9000000101"          # demo CSP Ramesh Kumar (RM 1)
SILENCE_WAIT_S = 1.2           # the call console's end-of-speech window (static/call.html)
EXACT_SLOTS = {"amount", "txn_id", "utr_or_ref"}   # numbers and IDs must match exactly
LOOSE_SLOTS = {"bank_name", "payment_mode", "month", "product"}
SLOTS = [*EXACT_SLOTS, *LOOSE_SLOTS]
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# ---------------------------------------------------------------- scoring helpers
def normalize(text: str) -> str:
    text = (text or "").translate(DEVANAGARI_DIGITS).lower()
    text = re.sub(r"<\|[^|]*\|>", " ", text)                 # Whisper special tokens like <|hi|>
    text = re.sub(r"[^\w\sऀ-ॿ]", " ", text)                  # drop punctuation, keep Devanagari
    return re.sub(r"\s+", " ", text).strip()


def _edits(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float | None:
    r, h = normalize(ref).split(), normalize(hyp).split()
    return round(_edits(r, h) / len(r), 3) if r else None


def cer(ref: str, hyp: str) -> float | None:
    r, h = list(normalize(ref).replace(" ", "")), list(normalize(hyp).replace(" ", ""))
    return round(_edits(r, h) / len(r), 3) if r else None


def slot_match(slot: str, expected: str, got: str | None) -> bool:
    if not got:
        return False
    if slot == "amount":
        return re.sub(r"\D", "", got.translate(DEVANAGARI_DIGITS)) == re.sub(r"\D", "", expected)
    if slot in EXACT_SLOTS:
        return re.sub(r"[^A-Z0-9]", "", got.upper()) == re.sub(r"[^A-Z0-9]", "", expected.upper())
    e, g = normalize(expected), normalize(got)
    return bool(e) and (e in g or g in e)


def pct(hits: list[bool]) -> float | None:
    return round(100 * sum(hits) / len(hits), 1) if hits else None


def pctl(values: list[float], q: float) -> float | None:
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 3) if v else None


# ---------------------------------------------------------------- running one case
def understand(text: str, lang: str, engine: str, retries: int = 3) -> dict:
    """The agent's first turn on `text`, in a fresh call. Retries if Groq's free tier rate-limits us,
    so a rate limit is never scored as the agent's mistake."""
    for attempt in range(retries + 1):
        agent.force_engine("rules" if engine == "rules" else None)
        try:
            cid = agent.start_call("inbound", CALLER, lang=lang)["call_id"]
            res = agent.handle_turn(cid, text, None)
        finally:
            agent.force_engine(None)
        turn = db.q("SELECT engine, model, latency_ms, error FROM turns WHERE call_id=? AND role='csp' ORDER BY id DESC LIMIT 1",
                    (cid,), one=True) or {}
        err = turn.get("error") or ""
        if "RateLimit" in err and attempt < retries:
            wait = 25 * (attempt + 1)
            print(f"    rate-limited, waiting {wait}s and retrying…", flush=True)
            time.sleep(wait)
            continue
        call = agent.public_call(cid)
        return {"reply": res.get("reply", ""), "intent": call.get("intent"), "slots": call.get("slots") or {},
                "action": call.get("outcome") if res.get("ended") else "continue",
                "engine": turn.get("engine"), "model": turn.get("model"), "llm_ms": turn.get("latency_ms"),
                "error": err or None}
    raise AssertionError("unreachable")


def score(case: dict, out: dict) -> dict:
    s = {}
    if case.get("expected_intent"):
        s["intent_ok"] = out["intent"] == case["expected_intent"]
    if case.get("expected_action"):
        s["action_ok"] = out["action"] == case["expected_action"]
    for slot in SLOTS:
        if case.get(slot):
            s[f"slot_{slot}"] = slot_match(slot, case[slot], out["slots"].get(slot))
    return s


def run_case(case: dict, args, synth_dir: str) -> dict:
    lang = case.get("lang") or "hi"
    tts_lang = "en-IN" if lang == "en" else "hi-IN"
    row = {"id": case["id"], "lang": lang, "audio": case.get("audio") or "", "synthetic": False}
    ref = (case.get("reference_text") or "").strip()

    audio_path = os.path.join(EVALS, "audio", case["audio"]) if case.get("audio") else None
    if not audio_path and args.synthesize and ref:
        audio_path = os.path.join(synth_dir, f"{case['id']}.mp3")
        with open(audio_path, "wb") as f:
            f.write(asyncio.run(tts.synthesize(ref, tts_lang)))
        row.update(audio=os.path.relpath(audio_path, ROOT), synthetic=True)

    # 1. Hearing
    heard = None
    if audio_path and not args.no_stt:
        if not stt.available():
            raise SystemExit("Speech-to-text needs GROQ_API_KEY in .env (or run with --no-stt)")
        with open(audio_path, "rb") as f:
            data = f.read()
        t0 = time.perf_counter()
        heard = stt.transcribe(data, os.path.basename(audio_path), tts_lang)
        row["stt_s"] = round(time.perf_counter() - t0, 3)
        row["heard"] = heard
        row["stray_token"] = bool(re.search(r"<\|[^|]*\|>", heard))
        if ref:
            row["wer"], row["cer"] = wer(ref, heard), cer(ref, heard)

    # 2 + 3. Understanding: from what was heard, and from the correct text
    for label, text in (("heard", heard), ("ref", ref if not args.audio_only else None)):
        if not text:
            continue
        out = understand(text, lang, args.engine)
        row[f"{label}_intent"], row[f"{label}_action"] = out["intent"], out["action"]
        row[f"{label}_slots"] = json.dumps(out["slots"], ensure_ascii=False)
        row[f"{label}_engine"], row[f"{label}_llm_s"] = out["engine"], (out["llm_ms"] or 0) / 1000
        row[f"{label}_model"] = out["model"] or out["engine"]
        row[f"{label}_error"], row[f"{label}_reply"] = out["error"], out["reply"]
        for k, v in score(case, out).items():
            row[f"{label}_{k}"] = v
        if args.pace:
            time.sleep(args.pace)

    # 4. Speaking the reply
    reply = row.get("heard_reply") or row.get("ref_reply")
    if reply and not args.no_tts:
        t0 = time.perf_counter()
        asyncio.run(tts.synthesize(reply, tts_lang))
        row["tts_s"] = round(time.perf_counter() - t0, 3)
    if "stt_s" in row and row.get("heard_llm_s") is not None and "tts_s" in row:
        row["turn_s"] = round(SILENCE_WAIT_S + row["stt_s"] + row["heard_llm_s"] + row["tts_s"], 3)
    return row


# ---------------------------------------------------------------- report
def summarize(rows: list[dict]) -> dict:
    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    def acc(prefix):
        out = {"intent_pct": pct(col(f"{prefix}_intent_ok")), "action_pct": pct(col(f"{prefix}_action_ok"))}
        slot_hits = [r[k] for r in rows for k in r if k.startswith(f"{prefix}_slot_")]
        out["details_pct"] = pct(slot_hits)
        out["cases"] = sum(1 for r in rows if f"{prefix}_action" in r)
        out["fallback_turns"] = sum(1 for r in rows if r.get(f"{prefix}_error"))
        out["answered_by"] = dict(Counter(r[f"{prefix}_model"] for r in rows if r.get(f"{prefix}_model")))
        return out

    timing = {k: {"p50": pctl(col(k), .5), "p95": pctl(col(k), .95), "n": len(col(k))}
              for k in ("stt_s", "heard_llm_s", "ref_llm_s", "tts_s", "turn_s")}
    return {
        "cases": len(rows),
        "with_audio": sum(1 for r in rows if "heard" in r),
        "synthetic_audio": sum(1 for r in rows if r.get("synthetic")),
        "hearing": {"wer_mean": round(statistics.mean(col("wer")), 3) if col("wer") else None,
                    "cer_mean": round(statistics.mean(col("cer")), 3) if col("cer") else None,
                    "stray_tokens": sum(1 for r in rows if r.get("stray_token"))},
        "understanding_from_audio": acc("heard"),
        "understanding_from_text": acc("ref"),
        "timing_s": timing,
    }


def markdown(summary: dict, rows: list[dict], args) -> str:
    h, a, t, tm = summary["hearing"], summary["understanding_from_audio"], summary["understanding_from_text"], \
        summary["timing_s"]
    f = lambda v, unit="": "–" if v is None else f"{v}{unit}"  # noqa: E731
    by = lambda s: ", ".join(f"{m.split('/')[-1]} {n}" for m, n in sorted(s["answered_by"].items(), key=lambda x: -x[1])) or "–"  # noqa: E731
    lines = [f"# Eko Sahayak scorecard · {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Engine: **{agent.engine_mode() if args.engine != 'rules' else 'rules'}** · cases: {summary['cases']} · "
             f"with audio: {summary['with_audio']} (synthetic: {summary['synthetic_audio']})", ""]
    if summary["synthetic_audio"]:
        lines += ["> Synthetic audio is clean studio-quality speech from the Edge voice. Real CSP recordings will "
                  "score worse; use these numbers only to check the pipeline works.", ""]
    lines += ["| | From audio (what Whisper heard) | From correct text |", "|---|---|---|",
              f"| Issue type correct | {f(a['intent_pct'], '%')} | {f(t['intent_pct'], '%')} |",
              f"| Next action correct | {f(a['action_pct'], '%')} | {f(t['action_pct'], '%')} |",
              f"| Details correct (amount, IDs…) | {f(a['details_pct'], '%')} | {f(t['details_pct'], '%')} |",
              f"| Turns that fell back to rules | {a['fallback_turns']} | {t['fallback_turns']} |",
              f"| Answered by | {by(a)} | {by(t)} |", "",
              f"Hearing: word error rate **{f(h['wer_mean'])}**, character error rate **{f(h['cer_mean'])}**, "
              f"stray Whisper tokens in {h['stray_tokens']} case(s). Word error rate is high whenever the reference "
              "and Whisper use different scripts (Roman vs Devanagari); the character rate and the "
              "understanding scores are the fairer measures.", "",
              "| Time per turn (s) | p50 | p95 |", "|---|---|---|",
              f"| Speech to text | {f(tm['stt_s']['p50'])} | {f(tm['stt_s']['p95'])} |",
              f"| LLM reply | {f(tm['heard_llm_s']['p50'] or tm['ref_llm_s']['p50'])} | "
              f"{f(tm['heard_llm_s']['p95'] or tm['ref_llm_s']['p95'])} |",
              f"| Voice generation | {f(tm['tts_s']['p50'])} | {f(tm['tts_s']['p95'])} |",
              f"| **End of speech to agent voice** (incl. {SILENCE_WAIT_S}s silence wait) | "
              f"**{f(tm['turn_s']['p50'])}** | **{f(tm['turn_s']['p95'])}** |", "",
              "## Misses", ""]
    misses = []
    for r in rows:
        for p in ("heard", "ref"):
            bad = [k[len(p) + 1:-3] for k, v in r.items() if k.startswith(f"{p}_") and k.endswith("_ok") and v is False]
            bad += [k[len(p) + 6:] for k, v in r.items() if k.startswith(f"{p}_slot_") and v is False]
            if bad:
                src = "audio" if p == "heard" else "text"
                misses.append(f"- **{r['id']}** ({src}): wrong {', '.join(bad)} · got intent `{r.get(p + '_intent')}`, "
                              f"action `{r.get(p + '_action')}`, details {r.get(p + '_slots')}"
                              + (f" · heard: “{r['heard']}”" if p == "heard" else ""))
    lines += misses or ["None."]
    return "\n".join(lines) + "\n"


def load_cases(path: str, only: set[str] | None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        cases = [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    cases = [c for c in cases if c.get("id") and (not only or c["id"] in only)]
    for c in cases:
        if not c.get("reference_text") and not c.get("audio"):
            raise SystemExit(f"case {c['id']}: needs reference_text, audio, or both")
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=os.path.join(EVALS, "cases.csv"))
    ap.add_argument("--engine", choices=["auto", "rules"], default="auto",
                    help="auto = the configured LLM (Groq), rules = the offline fallback engine")
    ap.add_argument("--synthesize", action="store_true", help="make Edge-voice audio for cases with no recording")
    ap.add_argument("--audio-only", action="store_true", help="skip the correct-text comparison (halves LLM calls)")
    ap.add_argument("--no-stt", action="store_true", help="text only: skip speech-to-text")
    ap.add_argument("--no-tts", action="store_true", help="skip timing the reply voice")
    ap.add_argument("--pace", type=float, default=0, help="seconds to wait between LLM calls (free-tier limits)")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--out", default=os.path.join(EVALS, "results"))
    args = ap.parse_args(argv)

    db.reset_connection(os.environ["EKO_DB_PATH"])
    seed()
    cases = load_cases(args.cases, set(args.only.split(",")) if args.only else None)
    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    synth_dir = os.path.join(out_dir, "synthetic_audio")
    os.makedirs(synth_dir, exist_ok=True)

    print(f"Scoring {len(cases)} case(s) with engine={args.engine} → {os.path.relpath(out_dir, ROOT)}", flush=True)
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}", flush=True)
        try:
            rows.append(run_case(case, args, synth_dir))
        except SystemExit:
            raise
        except Exception as e:   # one broken file shouldn't sink the whole run
            print(f"    failed: {type(e).__name__}: {e}", flush=True)
            rows.append({"id": case["id"], "failed": f"{type(e).__name__}: {e}"})

    summary = summarize([r for r in rows if "failed" not in r])
    summary["failed_cases"] = [r["id"] for r in rows if "failed" in r]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "id", k))
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    report = markdown(summary, [r for r in rows if "failed" not in r], args)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    if not os.listdir(synth_dir):
        os.rmdir(synth_dir)
    print("\n" + report)
    if summary["failed_cases"]:
        print(f"Failed cases: {', '.join(summary['failed_cases'])}")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
