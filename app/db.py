"""SQLite storage. Swap for Postgres later by replacing this module."""
import json
import os
import sqlite3
import threading
import time

DB_PATH = os.environ.get("EKO_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "eko_agent.db"))
_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS rms (
    id INTEGER PRIMARY KEY, name TEXT, phone TEXT, region TEXT
);
CREATE TABLE IF NOT EXISTS csps (
    id INTEGER PRIMARY KEY, csp_code TEXT UNIQUE, name TEXT, shop_name TEXT, phone TEXT UNIQUE,
    district TEXT, rm_id INTEGER REFERENCES rms(id)
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    direction TEXT,            -- inbound | outbound
    purpose TEXT,              -- support | follow_up | activation | ...
    csp_id INTEGER, rm_id INTEGER, caller_phone TEXT,
    status TEXT,               -- ongoing | ended
    outcome TEXT,              -- ticket_raised | escalated | not_understood | resolved_by_agent | dropped | info_only
    intent TEXT, intent_confidence REAL, language TEXT, sentiment TEXT,
    slots_json TEXT DEFAULT '{}',
    misunderstood_streak INTEGER DEFAULT 0, misunderstood_total INTEGER DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    started_at REAL, ended_at REAL, duration_sec REAL,
    engine TEXT, summary TEXT, ticket_id INTEGER, escalation_reason TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY, call_id INTEGER, role TEXT, text TEXT, ts REAL,
    stt_confidence REAL, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
    engine TEXT, intent TEXT, intent_confidence REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY, ticket_no TEXT UNIQUE, call_id INTEGER, csp_id INTEGER, rm_id INTEGER,
    category TEXT,             -- category predicted by the agent
    rm_category TEXT,          -- category confirmed/corrected by the RM (ground truth for accuracy)
    priority TEXT, status TEXT, -- open | in_progress | resolved | closed
    escalated INTEGER DEFAULT 0, escalation_reason TEXT,
    summary TEXT, details_json TEXT DEFAULT '{}', slot_completeness REAL,
    created_at REAL, updated_at REAL, closed_at REAL, sla_due_at REAL
);
CREATE TABLE IF NOT EXISTS ticket_events (
    id INTEGER PRIMARY KEY, ticket_id INTEGER, ts REAL, actor TEXT, action TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, password_hash TEXT,
    rm_id INTEGER REFERENCES rms(id),   -- set when the user is one of the RMs
    role TEXT,                          -- admin | ops | rm | pending
    created_at REAL, last_login_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id), created_at REAL, expires_at REAL
);
CREATE TABLE IF NOT EXISTS upload_tokens (   -- one per RM phone running the uploader app (uploader/)
    id INTEGER PRIMARY KEY, token_hash TEXT UNIQUE, rm_id INTEGER REFERENCES rms(id), label TEXT,
    created_by TEXT, created_at REAL, last_used_at REAL, revoked_at REAL
);
CREATE TABLE IF NOT EXISTS recordings (      -- real RM-CSP call recordings from the Eko SIM (app/recordings.py)
    id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, rm_id INTEGER, token_id INTEGER, device_id TEXT,
    file_path TEXT, orig_name TEXT, size_bytes INTEGER,
    started_at REAL, duration_sec REAL, direction TEXT, other_number TEXT, csp_id INTEGER, uploaded_at REAL,
    status TEXT,               -- new | done | failed
    attempts INTEGER DEFAULT 0, error TEXT, processed_at REAL,
    is_csp_call INTEGER, language TEXT, transcript_json TEXT, issue_segment INTEGER,
    intent TEXT, intent_confidence REAL, slots_json TEXT, sentiment TEXT, resolution TEXT,
    rm_handling TEXT, summary TEXT, model TEXT
);
CREATE TABLE IF NOT EXISTS api_keys (         -- other systems calling the API (app/integrations.py)
    id INTEGER PRIMARY KEY, key_hash TEXT UNIQUE, name TEXT, role TEXT, created_by TEXT,
    created_at REAL, last_used_at REAL, revoked_at REAL
);
CREATE TABLE IF NOT EXISTS webhooks (         -- other systems told about events (app/integrations.py)
    id INTEGER PRIMARY KEY, url TEXT, events TEXT, secret TEXT, active INTEGER DEFAULT 1, created_by TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS webhook_deliveries (   -- outbox: one row per event per webhook
    id INTEGER PRIMARY KEY, webhook_id INTEGER, event TEXT, payload TEXT,
    status TEXT,               -- pending | delivered | failed
    attempts INTEGER DEFAULT 0, next_attempt_at REAL, last_status_code INTEGER, last_error TEXT,
    created_at REAL, delivered_at REAL
);
CREATE INDEX IF NOT EXISTS idx_deliveries_due ON webhook_deliveries(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status, uploaded_at);
CREATE INDEX IF NOT EXISTS idx_calls_started ON calls(started_at);
CREATE INDEX IF NOT EXISTS idx_turns_call ON turns(call_id);
CREATE INDEX IF NOT EXISTS idx_tickets_rm ON tickets(rm_id, status);
"""

_conn = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        # Migrations for databases created by older versions
        if "lang" not in {r[1] for r in _conn.execute("PRAGMA table_info(calls)")}:
            _conn.execute("ALTER TABLE calls ADD COLUMN lang TEXT")  # 'ask' | 'hi' | 'en'
            _conn.commit()
        if "touched_at" not in {r[1] for r in _conn.execute("PRAGMA table_info(calls)")}:
            # Real wall-clock time of the last activity (call `ts` can be in the past for simulated calls)
            _conn.execute("ALTER TABLE calls ADD COLUMN touched_at REAL")
            _conn.commit()
        if "model" not in {r[1] for r in _conn.execute("PRAGMA table_info(turns)")}:
            _conn.execute("ALTER TABLE turns ADD COLUMN model TEXT")   # which LLM answered this turn
            _conn.commit()
        if "next_try_at" not in {r[1] for r in _conn.execute("PRAGMA table_info(recordings)")}:
            _conn.execute("ALTER TABLE recordings ADD COLUMN next_try_at REAL")   # when a failed one is retried
            _conn.commit()
        if "simulated" not in {r[1] for r in _conn.execute("PRAGMA table_info(calls)")}:
            # Simulated calls send no webhook events. Every call so far was a demo or a console test
            _conn.execute("ALTER TABLE calls ADD COLUMN simulated INTEGER DEFAULT 0")
            _conn.execute("UPDATE calls SET simulated=1")
            _conn.commit()
        if "role" not in {r[1] for r in _conn.execute("PRAGMA table_info(users)")}:
            # Accounts made before roles existed: the first becomes admin, RM-linked ones RMs, the rest ops
            _conn.execute("ALTER TABLE users ADD COLUMN role TEXT")
            _conn.execute("""UPDATE users SET role = CASE WHEN id = (SELECT MIN(id) FROM users) THEN 'admin'
                             WHEN rm_id IS NOT NULL THEN 'rm' ELSE 'ops' END""")
            _conn.commit()
    return _conn


def reset_connection(path: str | None = None):
    """Used by tests to point at a fresh database."""
    global _conn, DB_PATH
    if _conn is not None:
        _conn.close()
    _conn = None
    if path:
        DB_PATH = path


def q(sql: str, args=(), one=False):
    with _lock:
        cur = conn().execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
    return (rows[0] if rows else None) if one else rows


def x(sql: str, args=()) -> int:
    with _lock:
        cur = conn().execute(sql, args)
        conn().commit()
        return cur.lastrowid


def update(table: str, row_id: int, **fields):
    if not fields:
        return
    for k, v in list(fields.items()):
        if isinstance(v, (dict, list)):
            fields[k] = json.dumps(v, ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in fields)
    x(f"UPDATE {table} SET {cols} WHERE id=?", (*fields.values(), row_id))


def now() -> float:
    return time.time()
