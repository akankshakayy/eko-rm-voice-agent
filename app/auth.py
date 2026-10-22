"""Email + password accounts, roles, and server-side sessions (SQLite).

Roles: admin (everything + user management), ops (everything except users),
rm (RM dashboard limited to their own CSPs), pending (new signup, no access until an
admin approves it). The first account ever created becomes admin.

Passwords are hashed with scrypt (stdlib). The browser gets a random session token in an
HttpOnly cookie; only its SHA-256 is stored, so a leaked database can't be replayed as logins.
"""
import hashlib
import hmac
import os
import re
import secrets
import threading
import time

from . import db

COOKIE = "eko_session"
SESSION_DAYS = 7
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"   # set to 1 when served over HTTPS
MIN_PASSWORD = 8
MAX_FAILS, LOCK_SECONDS = 5, 15 * 60

_N, _R, _P = 2 ** 14, 8, 1
_fails: dict[str, list[float]] = {}
_fails_lock = threading.Lock()


ROLES = ("admin", "ops", "rm", "pending")
ROLE_LABELS = {"admin": "Admin", "ops": "Ops", "rm": "RM", "pending": "Pending approval"}


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(dk) // 2)
        return hmac.compare_digest(got.hex(), dk)
    except (ValueError, TypeError):
        return False


_DUMMY_HASH = hash_password(secrets.token_hex(8))  # keeps login timing the same for unknown emails


# ---------------------------------------------------------------- accounts
def _clean_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 254:
        raise AuthError("Enter a valid email address")
    return email


def public_user(u: dict) -> dict:
    rm = db.q("SELECT name, region FROM rms WHERE id=?", (u["rm_id"],), one=True) if u.get("rm_id") else None
    return {"id": u["id"], "email": u["email"], "name": u["name"], "role": u.get("role") or "pending",
            "role_label": ROLE_LABELS.get(u.get("role"), "Pending approval"), "rm_id": u.get("rm_id"),
            "rm_name": rm["name"] if rm else None, "rm_region": rm["region"] if rm else None,
            "created_at": u.get("created_at"), "last_login_at": u.get("last_login_at")}


def signup(email: str, password: str, name: str, rm_id: int | None = None) -> dict:
    email = _clean_email(email)
    name = (name or "").strip()
    if not name or len(name) > 100:
        raise AuthError("Enter your name")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters")
    if len(password) > 256:
        raise AuthError("Password is too long")
    if rm_id is not None:
        if not db.q("SELECT id FROM rms WHERE id=?", (rm_id,), one=True):
            raise AuthError("Unknown RM profile")
        if db.q("SELECT id FROM users WHERE rm_id=?", (rm_id,), one=True):
            raise AuthError("That RM profile already has an account", 409)
    if db.q("SELECT id FROM users WHERE email=?", (email,), one=True):
        raise AuthError("An account with this email already exists", 409)
    first = not db.q("SELECT id FROM users LIMIT 1", one=True)
    role = "admin" if first else "pending"   # rm_id here is only a request until an admin approves it
    uid = db.x("INSERT INTO users(email,name,password_hash,rm_id,role,created_at) VALUES(?,?,?,?,?,?)",
               (email, name, hash_password(password), rm_id, role, time.time()))
    return db.q("SELECT * FROM users WHERE id=?", (uid,), one=True)


def set_role(actor: dict, user_id: int, role: str, rm_id: int | None) -> dict:
    """Admin changes a user's role / RM profile."""
    u = db.q("SELECT * FROM users WHERE id=?", (user_id,), one=True)
    if not u:
        raise AuthError("User not found", 404)
    if role not in ROLES:
        raise AuthError("Unknown role")
    if role == "rm":
        if rm_id is None or not db.q("SELECT id FROM rms WHERE id=?", (rm_id,), one=True):
            raise AuthError("Pick the RM profile for this user")
    if rm_id is not None and db.q("SELECT id FROM users WHERE rm_id=? AND id<>?", (rm_id, user_id), one=True):
        raise AuthError("That RM profile is already linked to another account", 409)
    if u["role"] == "admin" and role != "admin" and _admin_count() <= 1:
        raise AuthError("There must be at least one admin", 409)
    if user_id == actor["id"] and role != "admin":
        raise AuthError("You can't remove your own admin access", 409)
    db.update("users", user_id, role=role, rm_id=rm_id)
    return db.q("SELECT * FROM users WHERE id=?", (user_id,), one=True)


def delete_user(actor: dict, user_id: int):
    u = db.q("SELECT * FROM users WHERE id=?", (user_id,), one=True)
    if not u:
        raise AuthError("User not found", 404)
    if user_id == actor["id"]:
        raise AuthError("You can't delete your own account", 409)
    db.x("DELETE FROM sessions WHERE user_id=?", (user_id,))
    db.x("DELETE FROM users WHERE id=?", (user_id,))


def _admin_count() -> int:
    return db.q("SELECT COUNT(*) n FROM users WHERE role='admin'", one=True)["n"]


def _locked(email: str) -> bool:
    now = time.time()
    with _fails_lock:
        recent = [t for t in _fails.get(email, []) if now - t < LOCK_SECONDS]
        _fails[email] = recent
        return len(recent) >= MAX_FAILS


def login(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    if _locked(email):
        raise AuthError("Too many failed attempts. Try again in 15 minutes.", 429)
    u = db.q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if not verify_password(password or "", u["password_hash"] if u else _DUMMY_HASH) or not u:
        with _fails_lock:
            _fails.setdefault(email, []).append(time.time())
        raise AuthError("Wrong email or password", 401)
    with _fails_lock:
        _fails.pop(email, None)
    db.update("users", u["id"], last_login_at=time.time())
    return u


# ---------------------------------------------------------------- sessions
def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.x("DELETE FROM sessions WHERE expires_at < ?", (now,))
    db.x("INSERT INTO sessions(token_hash,user_id,created_at,expires_at) VALUES(?,?,?,?)",
         (_h(token), user_id, now, now + SESSION_DAYS * 86400))
    return token


def user_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    return db.q("""SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                   WHERE s.token_hash=? AND s.expires_at>?""", (_h(token), time.time()), one=True)


def end_session(token: str | None):
    if token:
        db.x("DELETE FROM sessions WHERE token_hash=?", (_h(token),))
