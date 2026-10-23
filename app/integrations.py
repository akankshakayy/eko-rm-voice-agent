"""Connecting Sahayak to other systems: Eko's CRM or ticketing tool, data pipelines, other teams' services.
The contract (what's stable, payloads, signatures) is in INTEGRATION.md.

In: service API keys. Another system calls the same REST API as the dashboards, with
    `Authorization: Bearer ekosk_...` instead of a login. A key acts with the Ops role, never Admin.
Out: webhooks. When something happens (ticket.created, ticket.updated, call.ended, recording.processed)
    Sahayak POSTs a signed JSON event to every subscribed URL.

Events go into an outbox table first, so a slow or broken receiver never holds up a live call, and nothing is
lost on a restart. A failed delivery is retried with backoff for about a day. Simulated calls send no events.

    python -m app.integrations key --name "Eko CRM"
    python -m app.integrations keys | revoke-key 2
    python -m app.integrations webhook --url https://crm.example/hooks/sahayak --events ticket.created,ticket.updated
    python -m app.integrations webhooks | remove-webhook 1 | test-webhook 1 | deliveries
"""
import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from urllib.parse import urlparse

from . import db

log = logging.getLogger("eko.integrations")
EVENTS = ("ticket.created", "ticket.updated", "call.ended", "recording.processed")
KEY_ROLE = "ops"
RETRY_AFTER = [30, 120, 600, 1800, 3600, 3 * 3600, 6 * 3600, 12 * 3600]   # ~23 h in all, then failed
SEND_TIMEOUT = 10
_wake = threading.Event()


def _h(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


# ---------------------------------------------------------------- service API keys
KEY_COLS = "id, name, role, created_by, created_at, last_used_at, revoked_at"


def create_key(name: str, created_by: str = "") -> dict:
    """The key itself is returned only here; the database keeps just its hash."""
    name = (name or "").strip()[:80]
    if not name:
        raise ValueError("Give the key a name, e.g. the system that will use it")
    key = "ekosk_" + secrets.token_urlsafe(32)
    kid = db.x("INSERT INTO api_keys(key_hash,name,role,created_by,created_at) VALUES(?,?,?,?,?)",
               (_h(key), name, KEY_ROLE, created_by, time.time()))
    return {**db.q(f"SELECT {KEY_COLS} FROM api_keys WHERE id=?", (kid,), one=True), "key": key}


def list_keys() -> list[dict]:
    return db.q(f"SELECT {KEY_COLS} FROM api_keys ORDER BY id DESC")


def revoke_key(key_id: int) -> bool:
    if not db.q("SELECT id FROM api_keys WHERE id=? AND revoked_at IS NULL", (key_id,), one=True):
        return False
    db.update("api_keys", key_id, revoked_at=time.time())
    return True


def service_for_key(key: str | None) -> dict | None:
    """The 'user' a request made with an API key acts as. Ticket history shows the key's name."""
    if not key:
        return None
    k = db.q("SELECT * FROM api_keys WHERE key_hash=? AND revoked_at IS NULL", (_h(key),), one=True)
    if not k:
        return None
    db.update("api_keys", k["id"], last_used_at=time.time())
    return {"id": None, "api_key_id": k["id"], "name": f"{k['name']} (API)", "email": None,
            "role": k["role"], "rm_id": None}


# ---------------------------------------------------------------- webhooks
HOOK_COLS = "id, url, events, active, created_by, created_at"


def _events(events) -> str:
    if isinstance(events, str):
        events = [e.strip() for e in events.split(",")]
    events = [e for e in (events or []) if e]
    if not events or "*" in events:
        return "*"
    bad = [e for e in events if e not in EVENTS]
    if bad:
        raise ValueError(f"Unknown event(s): {', '.join(bad)}. Choose from {', '.join(EVENTS)} or *")
    return ",".join(dict.fromkeys(events))


def create_webhook(url: str, events="*", created_by: str = "") -> dict:
    """The signing secret is returned only here; give it to the receiving system."""
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        raise ValueError("The webhook URL must start with https:// (or http:// inside Eko's network)")
    secret = "whsec_" + secrets.token_urlsafe(32)
    hid = db.x("INSERT INTO webhooks(url,events,secret,active,created_by,created_at) VALUES(?,?,?,1,?,?)",
               (u.geturl(), _events(events), secret, created_by, time.time()))
    return {**db.q(f"SELECT {HOOK_COLS} FROM webhooks WHERE id=?", (hid,), one=True), "secret": secret}


def list_webhooks() -> list[dict]:
    return db.q(f"""SELECT {HOOK_COLS},
                    (SELECT COUNT(*) FROM webhook_deliveries d WHERE d.webhook_id=w.id AND d.status='pending') pending,
                    (SELECT COUNT(*) FROM webhook_deliveries d WHERE d.webhook_id=w.id AND d.status='failed') failed,
                    (SELECT MAX(delivered_at) FROM webhook_deliveries d WHERE d.webhook_id=w.id) last_delivered_at
                    FROM webhooks w ORDER BY id DESC""")


def remove_webhook(hook_id: int) -> bool:
    """Stops future events. Its delivery history is kept; pending deliveries are dropped."""
    if not db.q("SELECT id FROM webhooks WHERE id=? AND active=1", (hook_id,), one=True):
        return False
    db.update("webhooks", hook_id, active=0)
    db.x("UPDATE webhook_deliveries SET status='failed', last_error='webhook removed' "
         "WHERE webhook_id=? AND status='pending'", (hook_id,))
    return True


def list_deliveries(hook_id: int | None = None, limit: int = 100) -> list[dict]:
    where = "WHERE webhook_id=?" if hook_id else ""
    return db.q(f"""SELECT id, webhook_id, event, status, attempts, last_status_code, last_error, created_at,
                    next_attempt_at, delivered_at FROM webhook_deliveries {where} ORDER BY id DESC LIMIT ?""",
                ((hook_id,) if hook_id else ()) + (max(1, min(limit, 1000)),))


def emit(event: str, data: dict, only_hook: int | None = None) -> int:
    """Queue an event for every webhook that wants it. Only writes to the outbox: never slows the caller."""
    hooks = db.q("SELECT id, events FROM webhooks WHERE active=1" + (" AND id=?" if only_hook else ""),
                 (only_hook,) if only_hook else ())
    hooks = [h for h in hooks if h["events"] == "*" or event in h["events"].split(",")]
    if not hooks:
        return 0
    payload = json.dumps({"event": event, "created_at": time.time(), "data": data}, ensure_ascii=False, default=str)
    now = time.time()
    for h in hooks:
        db.x("""INSERT INTO webhook_deliveries(webhook_id,event,payload,status,attempts,next_attempt_at,created_at)
                VALUES(?,?,?,'pending',0,?,?)""", (h["id"], event, payload, now, now))
    _wake.set()
    return len(hooks)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """What receivers check: HMAC-SHA256 of "<timestamp>.<raw body>" with the webhook's secret."""
    return "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def _post(url: str, body: bytes, headers: dict) -> int:
    import httpx
    return httpx.post(url, content=body, headers=headers, timeout=SEND_TIMEOUT).status_code


def deliver_pending(send=_post, now: float | None = None, limit: int = 50) -> dict:
    """Send every delivery that is due. A 2xx answer counts as delivered; anything else is retried later."""
    now = now or time.time()
    due = db.q("""SELECT d.*, w.url, w.secret FROM webhook_deliveries d JOIN webhooks w ON w.id=d.webhook_id
                  WHERE d.status='pending' AND d.next_attempt_at<=? ORDER BY d.id LIMIT ?""", (now, limit))
    sent = failed = 0
    for d in due:
        body = d["payload"].encode()
        ts = str(int(time.time()))
        headers = {"Content-Type": "application/json", "User-Agent": "Eko-Sahayak-Webhooks/1",
                   "X-Sahayak-Event": d["event"], "X-Sahayak-Delivery": str(d["id"]),
                   "X-Sahayak-Timestamp": ts, "X-Sahayak-Signature": sign(d["secret"], ts, body)}
        code, error = None, None
        try:
            code = send(d["url"], body, headers)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:300]
        attempts = d["attempts"] + 1
        if code is not None and 200 <= code < 300:
            db.update("webhook_deliveries", d["id"], status="delivered", attempts=attempts, last_status_code=code,
                      last_error=None, delivered_at=time.time())
            sent += 1
            continue
        gave_up = attempts > len(RETRY_AFTER)
        db.update("webhook_deliveries", d["id"], attempts=attempts, last_status_code=code,
                  last_error=error or f"HTTP {code}", status="failed" if gave_up else "pending",
                  next_attempt_at=None if gave_up else now + RETRY_AFTER[attempts - 1])
        failed += gave_up
    return {"due": len(due), "delivered": sent, "gave_up": failed}


def delivery_loop(stop: threading.Event):
    """Background thread (started by app.main): send due deliveries as soon as they are queued."""
    while not stop.is_set():
        try:
            deliver_pending()
        except Exception:
            log.exception("Webhook delivery failed")
        _wake.wait(15)
        _wake.clear()


def wake():
    _wake.set()


# ---------------------------------------------------------------- event payloads (stable: see INTEGRATION.md)
def _simulated(call_id) -> bool:
    c = db.q("SELECT simulated FROM calls WHERE id=?", (call_id,), one=True) if call_id else None
    return bool(c and c["simulated"])


def ticket_data(ticket_id: int) -> dict | None:
    t = db.q("""SELECT t.*, s.csp_code, s.name csp_name, s.shop_name, s.phone csp_phone, s.district, r.name rm_name
                FROM tickets t LEFT JOIN csps s ON s.id=t.csp_id LEFT JOIN rms r ON r.id=t.rm_id WHERE t.id=?""",
             (ticket_id,), one=True)
    if not t:
        return None
    return {"id": t["id"], "ticket_no": t["ticket_no"], "call_id": t["call_id"], "category": t["category"],
            "rm_category": t["rm_category"], "priority": t["priority"], "status": t["status"],
            "escalated": bool(t["escalated"]), "escalation_reason": t["escalation_reason"], "summary": t["summary"],
            "details": json.loads(t["details_json"] or "{}"), "slot_completeness": t["slot_completeness"],
            "created_at": t["created_at"], "updated_at": t["updated_at"], "closed_at": t["closed_at"],
            "sla_due_at": t["sla_due_at"],
            "csp": {"id": t["csp_id"], "code": t["csp_code"], "name": t["csp_name"], "shop": t["shop_name"],
                    "phone": t["csp_phone"], "district": t["district"]} if t["csp_id"] else None,
            "rm": {"id": t["rm_id"], "name": t["rm_name"]} if t["rm_id"] else None}


def call_data(call_id: int) -> dict:
    from .agent import public_call
    c = public_call(call_id)
    keep = ("id", "direction", "purpose", "caller_phone", "status", "outcome", "intent", "intent_confidence",
            "language", "lang", "sentiment", "slots", "turn_count", "started_at", "ended_at", "duration_sec",
            "engine", "summary", "ticket_id", "ticket_no", "escalation_reason", "csp_id", "csp_code", "csp_name",
            "rm_id", "rm_name")
    out = {k: c.get(k) for k in keep}
    out["transcript"] = db.q("SELECT role, text, ts FROM turns WHERE call_id=? ORDER BY id", (call_id,))
    return out


def _safe(fn):
    """Integration hooks must never break a call, a ticket update or a processing run."""
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            log.exception("Couldn't queue %s event", fn.__name__)
    return wrapped


@_safe
def ticket_created(ticket_id: int):
    t = ticket_data(ticket_id)
    if t and not _simulated(t["call_id"]):
        emit("ticket.created", t)


@_safe
def ticket_updated(ticket_id: int, changes: list[str], actor: str):
    t = ticket_data(ticket_id)
    if t and changes and not _simulated(t["call_id"]):
        emit("ticket.updated", {**t, "changes": changes, "changed_by": actor})


@_safe
def call_ended(call_id: int):
    if not _simulated(call_id):
        emit("call.ended", call_data(call_id))


@_safe
def recording_processed(rec_id: int):
    from .recordings import get_recording
    r = get_recording(rec_id)
    if r:
        emit("recording.processed", r)


def send_test(hook_id: int) -> int:
    return emit("ticket.created", {"test": True, "message": "Test event from Eko Sahayak"}, only_hook=hook_id)


# ---------------------------------------------------------------- command line
def _cli():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    p = argparse.ArgumentParser(prog="python -m app.integrations", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("key", help="make a service API key")
    k.add_argument("--name", required=True)
    sub.add_parser("keys", help="list service API keys")
    rk = sub.add_parser("revoke-key")
    rk.add_argument("id", type=int)
    w = sub.add_parser("webhook", help="subscribe a URL to events")
    w.add_argument("--url", required=True)
    w.add_argument("--events", default="*", help=f"comma-separated: {','.join(EVENTS)} (default: all)")
    sub.add_parser("webhooks", help="list webhooks")
    rw = sub.add_parser("remove-webhook")
    rw.add_argument("id", type=int)
    tw = sub.add_parser("test-webhook", help="queue a test event and send it now")
    tw.add_argument("id", type=int)
    sub.add_parser("deliveries", help="recent webhook deliveries")
    a = p.parse_args()
    if a.cmd == "key":
        r = create_key(a.name, "cli")
        print(f"API key for {r['name']}. Shown only once:\n\n  {r['key']}\n")
    elif a.cmd == "keys":
        for r in list_keys():
            print(f"{r['id']:>3}  {'revoked' if r['revoked_at'] else 'active':7}  {r['name']}")
    elif a.cmd == "revoke-key":
        print("Revoked" if revoke_key(a.id) else "No active key with that id")
    elif a.cmd == "webhook":
        r = create_webhook(a.url, a.events, "cli")
        print(f"Webhook {r['id']} -> {r['url']} ({r['events']}). Signing secret, shown only once:\n\n  {r['secret']}\n")
    elif a.cmd == "webhooks":
        for r in list_webhooks():
            print(f"{r['id']:>3}  {'active' if r['active'] else 'removed':7}  {r['url']}  [{r['events']}]  "
                  f"pending {r['pending']}, failed {r['failed']}")
    elif a.cmd == "remove-webhook":
        print("Removed" if remove_webhook(a.id) else "No active webhook with that id")
    elif a.cmd == "test-webhook":
        if not send_test(a.id):
            print("No active webhook with that id")
            return
        print(deliver_pending(), list_deliveries(a.id, 1))
    elif a.cmd == "deliveries":
        for d in list_deliveries():
            print(f"{d['id']:>5}  hook {d['webhook_id']}  {d['event']:<20} {d['status']:<9} tries {d['attempts']}  "
                  f"{d['last_error'] or d['last_status_code'] or ''}")


if __name__ == "__main__":
    _cli()
