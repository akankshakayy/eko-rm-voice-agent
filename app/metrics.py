"""Aggregations for the dashboards."""
import os
import statistics
from collections import Counter

from . import db
from .taxonomy import STT_CONFIDENCE_MIN, INTENT_CONFIDENCE_MIN, label

PRICE_IN = float(os.environ.get("LLM_PRICE_INPUT_PER_M", "1.0"))    # USD per 1M input tokens
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUTPUT_PER_M", "5.0"))  # USD per 1M output tokens
USD_INR = float(os.environ.get("USD_INR", "84"))


def _pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


def _p(values, q):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(q * (len(values) - 1))))
    return round(values[k], 1)


def agent_metrics(hours: float = 24 * 7, rm_id: int | None = None):
    since = db.now() - hours * 3600
    rm_sql, rm_args = ("AND c.rm_id=?", (rm_id,)) if rm_id else ("", ())
    calls = db.q(f"""SELECT c.*, s.name csp_name, s.csp_code, r.name rm_name, t.ticket_no, t.status ticket_status,
                     t.closed_at ticket_closed_at
                     FROM calls c LEFT JOIN csps s ON s.id=c.csp_id LEFT JOIN rms r ON r.id=c.rm_id
                     LEFT JOIN tickets t ON t.id=c.ticket_id
                     WHERE c.started_at>=? {rm_sql} ORDER BY c.started_at DESC""", (since, *rm_args))
    ended = [c for c in calls if c["status"] == "ended"]
    live = db.q(f"""SELECT c.id, c.direction, c.started_at, c.intent, c.turn_count, s.name csp_name, s.csp_code,
                    r.name rm_name FROM calls c LEFT JOIN csps s ON s.id=c.csp_id LEFT JOIN rms r ON r.id=c.rm_id
                    WHERE c.status='ongoing' {rm_sql} ORDER BY c.started_at""", rm_args)
    durations = [c["duration_sec"] for c in ended if c["duration_sec"] is not None]
    outcomes = Counter(c["outcome"] for c in ended)
    intents = Counter(c["intent"] or "unknown" for c in calls)

    # time series
    bucket = 3600 if hours <= 48 else 86400
    series = Counter()
    for c in calls:
        series[int(c["started_at"] // bucket * bucket)] += 1
    start = int(since // bucket * bucket)
    timeline = [{"t": t, "calls": series.get(t, 0)} for t in range(start, int(db.now()) + 1, bucket)]

    tickets = db.q(f"""SELECT * FROM tickets t WHERE created_at>=? {rm_sql.replace('c.', 't.')}""", (since, *rm_args))
    tstatus = Counter(t["status"] for t in tickets)
    res_hours = [(t["closed_at"] - t["created_at"]) / 3600 for t in tickets if t["closed_at"]]
    now = db.now()
    breached = sum(1 for t in tickets if (t["closed_at"] or now) > t["sla_due_at"])

    dur_buckets = [("<30s", 0, 30), ("30-60s", 30, 60), ("1-2m", 60, 120), ("2-3m", 120, 180),
                   ("3-5m", 180, 300), (">5m", 300, 1e9)]
    hist = [{"bucket": n, "count": sum(1 for d in durations if lo <= d < hi)} for n, lo, hi in dur_buckets]

    n_end = len(ended)
    handled_by_agent = outcomes["ticket_raised"] + outcomes["resolved_by_agent"] + outcomes["info_only"]
    return {
        "window_hours": hours,
        "totals": {
            "calls": len(calls), "inbound": sum(c["direction"] == "inbound" for c in calls),
            "outbound": sum(c["direction"] == "outbound" for c in calls), "ongoing": len(live), "ended": n_end,
            "avg_duration_sec": round(statistics.mean(durations), 1) if durations else 0,
            "median_duration_sec": round(statistics.median(durations), 1) if durations else 0,
            "ticket_raise_rate": _pct(sum(1 for c in ended if c["ticket_id"]), n_end),
            "escalation_rate": _pct(outcomes["escalated"], n_end),
            "not_understood_rate": _pct(outcomes["not_understood"], n_end),
            "agent_handled_rate": _pct(handled_by_agent, n_end),
            "self_resolved": outcomes["resolved_by_agent"],
        },
        "outcomes": dict(outcomes),
        "intents": [{"intent": k, "label": label(k) if k != "unknown" else "Unknown", "count": v}
                    for k, v in intents.most_common()],
        "timeline": timeline, "bucket_sec": bucket,
        "duration_hist": hist,
        "tickets": {
            "total": len(tickets), "open": tstatus["open"], "in_progress": tstatus["in_progress"],
            "resolved": tstatus["resolved"], "closed": tstatus["closed"],
            "escalated": sum(t["escalated"] for t in tickets),
            "avg_resolution_hours": round(statistics.mean(res_hours), 1) if res_hours else None,
            "sla_breached": breached,
        },
        "live_calls": live,
        "recent_calls": [{k: c[k] for k in ("id", "direction", "purpose", "started_at", "duration_sec", "status",
                                            "outcome", "intent", "intent_confidence", "csp_name", "csp_code",
                                            "rm_name", "ticket_no", "ticket_status", "ticket_closed_at",
                                            "escalation_reason", "turn_count", "engine", "sentiment")}
                         for c in calls[:60]],
    }


def model_metrics(hours: float = 24 * 7):
    since = db.now() - hours * 3600
    turns = db.q("""SELECT t.* FROM turns t JOIN calls c ON c.id=t.call_id WHERE c.started_at>=? AND t.role='csp'""",
                 (since,))
    nlu = [t for t in turns if t["engine"] not in ("guard", "system", None)]
    llm = [t for t in nlu if t["engine"] != "rules"]
    lat = [t["latency_ms"] for t in nlu if t["latency_ms"] is not None]
    llm_lat = [t["latency_ms"] for t in llm if t["latency_ms"] is not None]
    stt = [t["stt_confidence"] for t in turns if t["stt_confidence"] is not None]
    conf = [t["intent_confidence"] for t in nlu if t["intent_confidence"] is not None]
    tin = sum(t["input_tokens"] or 0 for t in turns)
    tout = sum(t["output_tokens"] or 0 for t in turns)
    paid = [t for t in turns if t["engine"] == "claude"]  # Groq / Gemini free tiers cost nothing
    cost_usd = (sum(t["input_tokens"] or 0 for t in paid) / 1e6 * PRICE_IN
                + sum(t["output_tokens"] or 0 for t in paid) / 1e6 * PRICE_OUT)
    errors = [t for t in turns if t["error"]]

    calls = db.q("SELECT * FROM calls WHERE started_at>=? AND status='ended'", (since,))
    tickets = db.q("SELECT category, rm_category, slot_completeness FROM tickets WHERE created_at>=?", (since,))
    labelled = [t for t in tickets if t["rm_category"]]
    correct = sum(1 for t in labelled if t["rm_category"] == t["category"])
    confusions = Counter((t["category"], t["rm_category"]) for t in labelled if t["rm_category"] != t["category"])
    per_intent = {}
    for t in labelled:
        d = per_intent.setdefault(t["rm_category"], {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += t["rm_category"] == t["category"]
    completeness = [t["slot_completeness"] for t in tickets if t["slot_completeness"] is not None]
    misunderstood = sum(c["misunderstood_total"] or 0 for c in calls)

    # latency per day for trend
    lat_series = {}
    for t in nlu:
        if t["latency_ms"] is None:
            continue
        day = int(t["ts"] // 86400 * 86400)
        lat_series.setdefault(day, []).append(t["latency_ms"])
    lat_trend = [{"t": d, "p50": _p(v, .5), "p95": _p(v, .95)} for d, v in sorted(lat_series.items())]

    return {
        "engine_mix": dict(Counter(t["engine"] for t in turns)),
        "nlu_turns": len(nlu),
        "latency_ms": {"p50": _p(lat, .5), "p95": _p(lat, .95), "max": _p(lat, 1.0),
                       "llm_p50": _p(llm_lat, .5), "llm_p95": _p(llm_lat, .95)},
        "latency_trend": lat_trend,
        "tokens": {"input": tin, "output": tout,
                   "avg_per_call": round((tin + tout) / len(calls)) if calls else 0},
        "cost": {"usd": round(cost_usd, 4), "inr": round(cost_usd * USD_INR, 2),
                 "inr_per_call": round(cost_usd * USD_INR / len(calls), 3) if calls else 0},
        "stt": {"avg_confidence": round(statistics.mean(stt), 3) if stt else None,
                "low_conf_pct": _pct(sum(1 for s in stt if s < STT_CONFIDENCE_MIN), len(stt)),
                "samples": len(stt)},
        "intent": {"avg_confidence": round(statistics.mean(conf), 3) if conf else None,
                   "low_conf_pct": _pct(sum(1 for c in conf if c < INTENT_CONFIDENCE_MIN), len(conf)),
                   "accuracy_pct": _pct(correct, len(labelled)) if labelled else None,
                   "labelled_tickets": len(labelled),
                   "per_intent": [{"intent": k, "label": label(k), "n": v["n"],
                                   "accuracy_pct": _pct(v["correct"], v["n"])} for k, v in per_intent.items()],
                   "top_confusions": [{"predicted": label(a), "actual": label(b), "count": n}
                                      for (a, b), n in confusions.most_common(5)]},
        "slot_completeness_pct": round(100 * statistics.mean(completeness), 1) if completeness else None,
        "misunderstood_turn_pct": _pct(misunderstood, len(turns)),
        "avg_turns_per_call": round(statistics.mean([c["turn_count"] for c in calls]), 1) if calls else 0,
        "errors": {"count": len(errors), "fallback_pct": _pct(len(errors), len(turns)),
                   "recent": [{"ts": e["ts"], "error": e["error"]} for e in errors[-5:]]},
    }
