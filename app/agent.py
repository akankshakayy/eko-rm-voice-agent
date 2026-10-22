"""Call orchestrator: owns call state, safety rules, ticketing and escalation.

The engine (an LLM or rules) only proposes what to say and do next. This
module decides what actually happens, so the escalation rules hold no matter
which engine is running.
"""
import json
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta

from . import db, i18n
from .seed import csp_by_phone
from .rules_engine import RulesEngine
from .taxonomy import (CATEGORIES, HUMAN_REQUEST, ANGER, STT_CONFIDENCE_MIN, INTENT_CONFIDENCE_MIN,
                       MAX_MISUNDERSTOOD, MAX_TURNS, label)

IST = timezone(timedelta(hours=5, minutes=30))
_rules = RulesEngine()
CALL_IDLE_TIMEOUT = float(os.environ.get("CALL_IDLE_TIMEOUT_MIN", "10")) * 60
_llm = {}
_local = threading.local()


def force_engine(mode: str | None):
    """Per-thread override (used by the simulator to stay on the free rules engine)."""
    _local.mode = mode


def engine_mode() -> str:
    """'rules', or the LLM provider in use: 'groq', 'gemini' or 'claude'."""
    if getattr(_local, "mode", None) == "rules":
        return "rules"
    if os.environ.get("AGENT_ENGINE", "auto").lower() == "rules":
        return "rules"
    from .llm_engine import provider
    return provider() or "rules"


def _engine():
    mode = engine_mode()
    if mode == "rules":
        return _rules
    if mode not in _llm:
        from .llm_engine import make_engine
        _llm[mode] = make_engine(mode)
    return _llm[mode]


# ---------------------------------------------------------------- helpers
def _call(call_id):
    c = db.q("SELECT * FROM calls WHERE id=?", (call_id,), one=True)
    if not c:
        raise KeyError(f"call {call_id} not found")
    c["slots"] = json.loads(c.get("slots_json") or "{}")
    return c


def _log_turn(call_id, role, text, ts, **kw):
    db.x("""INSERT INTO turns(call_id,role,text,ts,stt_confidence,latency_ms,input_tokens,output_tokens,engine,
            intent,intent_confidence,error,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
         (call_id, role, text, ts, kw.get("stt_confidence"), kw.get("latency_ms"), kw.get("input_tokens"),
          kw.get("output_tokens"), kw.get("engine"), kw.get("intent"), kw.get("intent_confidence"), kw.get("error"),
          kw.get("model")))


def _has(text, words):
    t = text.lower()
    return any(w in t for w in words)


def _state(call):
    csp = db.q("SELECT * FROM csps WHERE id=?", (call["csp_id"],), one=True) if call["csp_id"] else None
    rm = db.q("SELECT * FROM rms WHERE id=?", (call["rm_id"],), one=True) if call["rm_id"] else None
    hist = db.q("SELECT role,text FROM turns WHERE call_id=? ORDER BY id", (call["id"],))
    tickets = db.q("SELECT ticket_no,category,status,summary FROM tickets WHERE csp_id=? ORDER BY id DESC LIMIT 5",
                   (call["csp_id"],)) if call["csp_id"] else []
    return {**call, "csp": csp, "rm_name": rm["name"] if rm else None, "history": hist, "csp_tickets": tickets}


def _greeting(call, csp, rm, lang):
    if not csp:
        return i18n.t("greet_unknown", lang)
    name = csp["name"].split()[0]
    rm_name = rm["name"].split()[0] if rm else i18n.t("your_rm", lang)
    key = "greet_outbound" if call["direction"] == "outbound" else "greet_inbound"
    return i18n.t(key, lang, name=name, rm=rm_name)


def _people(call):
    csp = db.q("SELECT * FROM csps WHERE id=?", (call["csp_id"],), one=True) if call["csp_id"] else None
    rm = db.q("SELECT * FROM rms WHERE id=?", (call["rm_id"],), one=True) if call["rm_id"] else None
    return csp, rm


def _say(call_id, ts, text, reply, stt_confidence=None):
    """Log a CSP turn + a fixed agent reply and keep the call going."""
    _log_turn(call_id, "csp", text, ts, stt_confidence=stt_confidence, engine="system")
    _log_turn(call_id, "agent", reply, ts, engine="system")
    return {"call_id": call_id, "reply": reply, "ended": False, "call": public_call(call_id)}


# ---------------------------------------------------------------- public API
def start_call(direction="inbound", phone=None, purpose="support", ts=None, lang="hi"):
    """lang: 'hi' or 'en', or 'ask' to let the caller choose at the start of the call."""
    ts = ts or db.now()
    csp = csp_by_phone(phone) if phone else None
    rm = db.q("SELECT * FROM rms WHERE id=?", (csp["rm_id"],), one=True) if csp else None
    lang = lang if lang in ("hi", "en", "ask") else "hi"
    call_id = db.x("""INSERT INTO calls(direction,purpose,csp_id,rm_id,caller_phone,status,started_at,engine,slots_json,lang,
                      touched_at) VALUES(?,?,?,?,?,'ongoing',?,?,'{}',?,?)""",
                   (direction, purpose, csp["id"] if csp else None, rm["id"] if rm else None, phone, ts, engine_mode(),
                    lang, time.time()))
    call = _call(call_id)
    reply = i18n.t("choose_lang", "hi") if lang == "ask" else _greeting(call, csp, rm, lang)
    _log_turn(call_id, "agent", reply, ts, engine="system")
    return {"call_id": call_id, "reply": reply, "ended": False, "call": public_call(call_id)}


def handle_turn(call_id, text, stt_confidence=None, ts=None):
    ts = ts or db.now()
    call = _call(call_id)
    if call["status"] != "ongoing":
        return {"call_id": call_id, "reply": "", "ended": True, "call": public_call(call_id)}
    text = (text or "").strip()
    turn_count = call["turn_count"] + 1
    db.update("calls", call_id, turn_count=turn_count, touched_at=time.time())

    # 0. Language: the caller picks Hindi / English, and can switch mid-call
    lang = call.get("lang") or "hi"
    short = len(text.split()) <= 6
    if lang == "ask":
        picked = i18n.choice(text)
        lang = picked or i18n.guess(text)
        db.update("calls", call_id, lang=lang)
        call = _call(call_id)
        if picked and short:
            return _say(call_id, ts, text, _greeting(call, *_people(call), lang), stt_confidence)
    elif (sw := i18n.switch_request(text)) and sw != lang:
        lang = sw
        db.update("calls", call_id, lang=lang)
        call = _call(call_id)
        if short:
            return _say(call_id, ts, text, i18n.t("lang_switched", lang), stt_confidence)

    # 1. Identify an unknown caller by CSP code
    if not call["csp_id"]:
        m = re.search(r"c\s*s\s*p\s*([\d\s]{5,8})", text.lower())
        if m:
            code = "CSP" + re.sub(r"\s", "", m.group(1))
            csp = db.q("SELECT * FROM csps WHERE csp_code=?", (code,), one=True)
            if csp:
                db.update("calls", call_id, csp_id=csp["id"], rm_id=csp["rm_id"])
                return _say(call_id, ts, text, i18n.t("identified", lang, name=csp["name"].split()[0]), stt_confidence)

    # 2. Speech not understood (ASR confidence / empty)
    garbled = (not re.search(r"[A-Za-zऀ-ॿ]{2,}", text)) or \
              (stt_confidence is not None and stt_confidence < STT_CONFIDENCE_MIN)
    if garbled:
        _log_turn(call_id, "csp", text or "(silence)", ts, stt_confidence=stt_confidence, intent="unknown",
                  intent_confidence=0, engine="guard")
        return _misunderstood(call, ts, "Low speech-recognition confidence / unclear audio")

    # 3. Hard escalation triggers - checked before the engine so they always win
    if _has(text, HUMAN_REQUEST) or _has(text, ANGER):
        asked = _has(text, HUMAN_REQUEST)
        reason = "CSP asked to speak to RM" if asked else "CSP sounded angry / frustrated"
        _log_turn(call_id, "csp", text, ts, stt_confidence=stt_confidence, engine="guard")
        reply = i18n.t("human_escalate" if asked else "angry_escalate", lang)
        return _finish(call, ts, "escalate", reply, {"escalation_reason": reason,
                                                     "sentiment": "angry" if "angry" in reason else "neutral"})

    # 4. Ask the engine
    state = _state(call)
    eng = _engine()
    error = None
    try:
        d = eng.decide(state, text)
    except Exception as e:  # LLM down / timeout -> rules fallback for this turn
        error = f"{type(e).__name__}: {str(e)[:150]}"
        d = _rules.decide(state, text)
    _log_turn(call_id, "csp", text, ts, stt_confidence=stt_confidence, latency_ms=d.get("latency_ms"),
              input_tokens=d.get("input_tokens"), output_tokens=d.get("output_tokens"), engine=d["engine"],
              intent=d.get("intent"), intent_confidence=d.get("intent_confidence"), error=error, model=d.get("model"))

    db.update("calls", call_id, intent=d.get("intent"), intent_confidence=d.get("intent_confidence"),
              language=d.get("language"), sentiment=d.get("sentiment"), slots_json=d.get("slots") or {},
              summary=d.get("summary") or call["summary"])
    call = _call(call_id)

    if not d.get("understood", True):
        return _misunderstood(call, ts, "Agent could not understand the issue", reply=d.get("reply"))
    db.update("calls", call_id, misunderstood_streak=0)

    action = d.get("action", "continue")
    mandatory = CATEGORIES.get(d.get("intent"), {}).get("always_escalate") and \
        (d.get("intent_confidence") or 0) >= INTENT_CONFIDENCE_MIN
    if mandatory and action != "escalate":
        # Fraud goes straight to the RM, whatever the engine wanted to do next (ask more, raise a ticket)
        slots = {**(call["slots"] or {})}
        slots.setdefault("issue_detail", text[:200])
        db.update("calls", call_id, slots_json=slots)
        call = _call(call_id)
        action, d["escalation_reason"] = "escalate", "Suspected fraud - mandatory RM escalation"
        d["reply"] = i18n.t("fraud", lang)
    if d.get("sentiment") == "angry" and action != "escalate":
        # The keyword check above misses anger when a word is misheard ("बकवा" for "बकवास");
        # the engine still hears the tone, so an angry caller always reaches a person
        action, d["escalation_reason"] = "escalate", "CSP sounded angry (detected by agent)"
        d["reply"] = i18n.t("angry_escalate", lang)
    if action == "continue" and turn_count >= MAX_TURNS:
        action, d["escalation_reason"] = "escalate", "Call exceeded max turns without resolution"
        d["reply"] = i18n.t("max_turns", lang)

    if action == "continue":
        _log_turn(call_id, "agent", d["reply"], ts, engine=d["engine"])
        return {"call_id": call_id, "reply": d["reply"], "ended": False, "call": public_call(call_id)}
    return _finish(call, ts, action, d.get("reply", ""), d)


def hangup(call_id, ts=None):
    """CSP hung up. If an issue was partly captured, hand it to the RM so it isn't lost."""
    ts = ts or db.now()
    call = _call(call_id)
    if call["status"] != "ongoing":
        return public_call(call_id)
    if call["intent"] and call["intent"] not in ("unknown",) and \
            not CATEGORIES.get(call["intent"], {}).get("self_serve"):
        _finish(call, ts, "escalate", "", {"escalation_reason": "Call dropped before ticket was confirmed"},
                outcome="dropped")
    else:
        db.update("calls", call_id, status="ended", outcome="dropped", ended_at=ts,
                  duration_sec=round(ts - call["started_at"], 1))
    return public_call(call_id)


def close_stale_calls(idle: float = CALL_IDLE_TIMEOUT, now: float | None = None) -> list[int]:
    """Hang up calls with no activity for `idle` seconds (tab closed, network lost, client crashed).
    Same as a hang-up: a partly captured issue still reaches the RM as an escalated ticket."""
    now = now or time.time()
    stale = db.q("""SELECT c.id, COALESCE(c.touched_at, MAX(t.ts), c.started_at) touched,
                           COALESCE(MAX(t.ts), c.started_at) last_ts
                    FROM calls c LEFT JOIN turns t ON t.call_id=c.id
                    WHERE c.status='ongoing' GROUP BY c.id HAVING touched < ?""", (now - idle,))
    for c in stale:
        hangup(c["id"], ts=c["last_ts"])   # end the call at its last activity, not now
    return [c["id"] for c in stale]


# ---------------------------------------------------------------- internals
def _misunderstood(call, ts, reason, reply=None):
    streak = call["misunderstood_streak"] + 1
    db.update("calls", call["id"], misunderstood_streak=streak, misunderstood_total=call["misunderstood_total"] + 1)
    if streak >= MAX_MISUNDERSTOOD:
        reply = i18n.t("not_understood_final", call.get("lang"))
        return _finish(_call(call["id"]), ts, "escalate", reply,
                       {"escalation_reason": f"Not understood: {reason}"}, outcome="not_understood")
    reply = reply or i18n.t("not_understood_retry", call.get("lang"))
    _log_turn(call["id"], "agent", reply, ts, engine="guard")
    return {"call_id": call["id"], "reply": reply, "ended": False, "call": public_call(call["id"])}


def _finish(call, ts, action, reply, d, outcome=None):
    ticket = None
    rm = db.q("SELECT name FROM rms WHERE id=?", (call["rm_id"],), one=True) if call["rm_id"] else None
    lang = call.get("lang")
    rm_first = rm["name"].split()[0] if rm else i18n.t("rm_team", lang)

    if action == "create_ticket":
        ticket = _create_ticket(call, ts, escalated=False, summary=d.get("summary"))
        sla = CATEGORIES.get(call["intent"], {}).get("sla_hours", 24)
        done = i18n.t("ticket_raised", lang, no=_speak(ticket["ticket_no"]), rm=rm_first, sla=sla)
        reply = f"{reply} {done}".strip()
        outcome = outcome or "ticket_raised"
    elif action == "escalate":
        ticket = _create_ticket(call, ts, escalated=True, reason=d.get("escalation_reason"), summary=d.get("summary"))
        if reply:
            reply = f"{reply} {i18n.t('reference', lang, no=_speak(ticket['ticket_no']))}"
        outcome = outcome or "escalated"
    elif action == "resolved_by_agent":
        if not reply:
            reply = _ticket_status_reply(call)
        outcome = "resolved_by_agent"
    else:
        reply = reply or i18n.t("bye", lang)
        outcome = outcome or "info_only"

    if reply:
        _log_turn(call["id"], "agent", reply, ts, engine=d.get("engine", "system"))
    db.update("calls", call["id"], status="ended", outcome=outcome, ended_at=ts,
              duration_sec=round(ts - call["started_at"], 1), ticket_id=ticket["id"] if ticket else None,
              escalation_reason=d.get("escalation_reason"), sentiment=d.get("sentiment") or call.get("sentiment"))
    return {"call_id": call["id"], "reply": reply, "ended": True, "outcome": outcome, "ticket": ticket,
            "call": public_call(call["id"])}


def _speak(ticket_no):
    """Last part of the ticket number, digit by digit so the voice reads it clearly."""
    return " ".join(ticket_no.split("-")[-1])


def _ticket_status_reply(call):
    lang = call.get("lang")
    t = db.q("SELECT * FROM tickets WHERE csp_id=? ORDER BY id DESC LIMIT 1", (call["csp_id"],), one=True) \
        if call["csp_id"] else None
    if not t:
        return i18n.t("no_tickets", lang)
    st = i18n.t(f"st_{t['status']}", lang) if f"st_{t['status']}" in i18n.PHRASES else t["status"]
    cat = i18n.spoken(t["category"], label(t["category"]), lang)
    return i18n.t("ticket_status", lang, no=_speak(t["ticket_no"]), cat=cat, st=st)


def _create_ticket(call, ts, escalated, reason=None, summary=None):
    cat = call["intent"] if call["intent"] in CATEGORIES else "unclassified"
    cfg = CATEGORIES.get(cat, {})
    req = cfg.get("slots", [])
    slots = {k: v for k, v in (call["slots"] or {}).items() if not k.startswith("_")}
    completeness = round(sum(1 for s in req if slots.get(s)) / len(req), 2) if req else 1.0
    priority = cfg.get("priority", "medium")
    if escalated and priority in ("low", "medium"):
        priority = "high"
    sla_h = 4 if escalated and cat != "fraud_report" else cfg.get("sla_hours", 24)
    if not summary and cat == "unclassified":
        summary = f"Issue not identified - please call the CSP back. ({reason or 'escalated'})"
    if not summary:
        details = ", ".join(f"{k}={v}" for k, v in slots.items())
        summary = f"{label(cat)}" + (f" ({details})" if details else "")
    tid = db.x("""INSERT INTO tickets(call_id,csp_id,rm_id,category,priority,status,escalated,escalation_reason,
                  summary,details_json,slot_completeness,created_at,updated_at,sla_due_at)
                  VALUES(?,?,?,?,?,'open',?,?,?,?,?,?,?,?)""",
               (call["id"], call["csp_id"], call["rm_id"], cat, priority, 1 if escalated else 0, reason, summary,
                json.dumps(slots, ensure_ascii=False), completeness, ts, ts, ts + sla_h * 3600))
    no = f"EKO-{datetime.fromtimestamp(ts, IST):%y%m%d}-{tid:04d}"
    db.update("tickets", tid, ticket_no=no)
    db.x("INSERT INTO ticket_events(ticket_id,ts,actor,action,note) VALUES(?,?,?,?,?)",
         (tid, ts, "agent", "escalated" if escalated else "created", reason))
    return db.q("SELECT * FROM tickets WHERE id=?", (tid,), one=True)


def update_ticket(ticket_id, actor="rm", status=None, rm_category=None, note=None, ts=None):
    ts = ts or db.now()
    t = db.q("SELECT * FROM tickets WHERE id=?", (ticket_id,), one=True)
    if not t:
        raise KeyError(ticket_id)
    fields = {"updated_at": ts}
    if status and status != t["status"]:
        fields["status"] = status
        if status in ("resolved", "closed") and not t["closed_at"]:
            fields["closed_at"] = ts
        if status in ("open", "in_progress"):
            fields["closed_at"] = None
        db.x("INSERT INTO ticket_events(ticket_id,ts,actor,action,note) VALUES(?,?,?,?,?)",
             (ticket_id, ts, actor, f"status:{status}", None))
    if rm_category and rm_category != t["rm_category"]:
        fields["rm_category"] = rm_category
        db.x("INSERT INTO ticket_events(ticket_id,ts,actor,action,note) VALUES(?,?,?,?,?)",
             (ticket_id, ts, actor, "category_confirmed", f"{t['category']} -> {rm_category}"))
    if note:
        db.x("INSERT INTO ticket_events(ticket_id,ts,actor,action,note) VALUES(?,?,?,?,?)",
             (ticket_id, ts, actor, "note", note))
    db.update("tickets", ticket_id, **fields)
    return db.q("SELECT * FROM tickets WHERE id=?", (ticket_id,), one=True)


def public_call(call_id):
    c = db.q("""SELECT c.*, s.name csp_name, s.csp_code, s.shop_name, r.name rm_name, t.ticket_no, t.status ticket_status
                FROM calls c LEFT JOIN csps s ON s.id=c.csp_id LEFT JOIN rms r ON r.id=c.rm_id
                LEFT JOIN tickets t ON t.id=c.ticket_id WHERE c.id=?""", (call_id,), one=True)
    c["slots"] = {k: v for k, v in json.loads(c.pop("slots_json") or "{}").items() if not k.startswith("_")}
    return c
