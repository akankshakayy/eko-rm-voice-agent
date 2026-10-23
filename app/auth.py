"""Email + password accounts, roles, server-side sessions, password reset and email verification (SQLite).

Roles: admin (everything + user management), ops (everything except users),
rm (RM dashboard limited to their own CSPs), pending (new signup, no access until an
admin approves it). The first account ever created becomes admin.

Passwords are hashed with scrypt (stdlib). The browser gets a random session token in an
HttpOnly cookie; only its SHA-256 is stored, so a leaked database can't be replayed as logins.
Reset and verification links work the same way: one use, short-lived, only their hash is stored.
Sign-ins, failures, resets and role changes go to auth_events; the login lockout counts from there,
so it survives a restart.

Email linking: when someone verifies an email address that the calling sheet gives for an RM, their
account is linked to that RM profile (an admin still approves the account).
"""
import hashlib
import hmac
import os
import re
import secrets
import time

from . import db, mailer

COOKIE = "eko_session"
SESSION_DAYS = 7
SESSION_IDLE_HOURS = float(os.environ.get("SESSION_IDLE_HOURS", "12"))   # signed out after this long unused
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"   # set to 1 when served over HTTPS
MIN_PASSWORD = 10
MAX_FAILS, LOCK_SECONDS = 5, 15 * 60
RESET_MINUTES, VERIFY_HOURS = 30, 48
MAX_RESETS_PER_HOUR = 3
DEFAULT_DOMAINS = "eko.co.in,ekosupport.in"

_N, _R, _P = 2 ** 14, 8, 1

ROLES = ("admin", "ops", "rm", "pending")
ROLE_LABELS = {"admin": "Admin", "ops": "Ops", "rm": "RM", "pending": "Pending approval"}

# The most common passwords that pass the length rule; rejected outright
COMMON = {"password12", "password123", "1234567890", "qwertyuiop", "iloveyou12", "eko@123456", "admin@1234",
          "password@123", "welcome@123", "india@1234", "12345678910", "qwerty@123", "abcd@12345", "sahayak@123"}


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


def check_password(password: str, email: str = "", name: str = ""):
    password = password or ""
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters")
    if len(password) > 256:
        raise AuthError("Password is too long")
    low = password.lower()
    if low in COMMON or len(set(low)) < 4:
        raise AuthError("That password is too easy to guess")
    words = [email.split("@")[0].lower()] + [w.lower() for w in (name or "").split()]
    if any(len(w) >= 4 and w in low for w in words):
        raise AuthError("Don't use your name or email in the password")


# ---------------------------------------------------------------- activity log
def log_event(event: str, user_id: int | None = None, email: str | None = None, ip: str | None = None,
              detail: str | None = None):
    db.x("INSERT INTO auth_events(ts,user_id,email,event,ip,detail) VALUES(?,?,?,?,?,?)",
         (time.time(), user_id, (email or "").lower() or None, event, ip, detail))


def activity(user_id: int, limit: int = 20) -> list[dict]:
    return db.q("SELECT ts, event, ip, detail FROM auth_events WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit))


# ---------------------------------------------------------------- accounts
def allowed_domains() -> list[str]:
    raw = os.environ.get("ALLOWED_EMAIL_DOMAINS", DEFAULT_DOMAINS).strip()
    return [] if raw in ("", "*") else [d.strip().lower().lstrip("@") for d in raw.split(",") if d.strip()]


def _clean_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 254:
        raise AuthError("Enter a valid email address")
    domains = allowed_domains()
    if domains and email.split("@")[1] not in domains:
        raise AuthError(f"Use your work email ({', '.join('@' + d for d in domains)})")
    return email


def public_user(u: dict) -> dict:
    rm = db.q("SELECT name, region FROM rms WHERE id=?", (u["rm_id"],), one=True) if u.get("rm_id") else None
    return {"id": u["id"], "email": u["email"], "name": u["name"], "role": u.get("role") or "pending",
            "role_label": ROLE_LABELS.get(u.get("role"), "Pending approval"), "rm_id": u.get("rm_id"),
            "rm_name": rm["name"] if rm else None, "rm_region": rm["region"] if rm else None,
            "email_verified": bool(u.get("email_verified_at")),
            "created_at": u.get("created_at"), "last_login_at": u.get("last_login_at")}


def _user(user_id: int) -> dict | None:
    return db.q("SELECT * FROM users WHERE id=?", (user_id,), one=True)


def signup(email: str, password: str, name: str, rm_id: int | None = None, ip: str | None = None) -> dict:
    email = _clean_email(email)
    name = (name or "").strip()
    if not name or len(name) > 100:
        raise AuthError("Enter your name")
    check_password(password, email, name)
    if rm_id is not None:
        if not db.q("SELECT id FROM rms WHERE id=?", (rm_id,), one=True):
            raise AuthError("Unknown RM profile")
        if db.q("SELECT id FROM users WHERE rm_id=?", (rm_id,), one=True):
            raise AuthError("That RM profile already has an account", 409)
    if db.q("SELECT id FROM users WHERE email=?", (email,), one=True):
        raise AuthError("An account with this email already exists", 409)
    first = not db.q("SELECT id FROM users LIMIT 1", one=True)
    role = "admin" if first else "pending"   # rm_id here is only a request until an admin approves it
    now = time.time()
    uid = db.x("INSERT INTO users(email,name,password_hash,rm_id,role,created_at,password_changed_at)"
               " VALUES(?,?,?,?,?,?,?)", (email, name, hash_password(password), rm_id, role, now, now))
    log_event("signup", uid, email, ip)
    u = _user(uid)
    send_verification(u)
    return u


def set_role(actor: dict, user_id: int, role: str, rm_id: int | None) -> dict:
    """Admin changes a user's role / RM profile."""
    u = _user(user_id)
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
    if (u["role"], u["rm_id"]) != (role, rm_id):
        log_event("role_changed", user_id, u["email"], detail=f"{u['role']} -> {role}, rm {rm_id} by {actor['email']}")
    return _user(user_id)


def delete_user(actor: dict, user_id: int):
    u = _user(user_id)
    if not u:
        raise AuthError("User not found", 404)
    if user_id == actor["id"]:
        raise AuthError("You can't delete your own account", 409)
    db.x("DELETE FROM sessions WHERE user_id=?", (user_id,))
    db.x("DELETE FROM auth_tokens WHERE user_id=?", (user_id,))
    db.x("DELETE FROM users WHERE id=?", (user_id,))
    log_event("user_deleted", user_id, u["email"], detail=f"by {actor['email']}")


def _admin_count() -> int:
    return db.q("SELECT COUNT(*) n FROM users WHERE role='admin'", one=True)["n"]


def _locked(email: str) -> bool:
    since = time.time() - LOCK_SECONDS
    last_ok = db.q("SELECT MAX(ts) t FROM auth_events WHERE email=? AND event IN ('login','password_reset')",
                   (email,), one=True)["t"] or 0
    n = db.q("SELECT COUNT(*) n FROM auth_events WHERE email=? AND event='login_failed' AND ts>?",
             (email, max(since, last_ok)), one=True)["n"]
    return n >= MAX_FAILS


def login(email: str, password: str, ip: str | None = None) -> dict:
    email = (email or "").strip().lower()[:254]
    if _locked(email):
        log_event("login_locked", email=email, ip=ip)
        raise AuthError("Too many failed attempts. Try again in 15 minutes, or reset your password.", 429)
    u = db.q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if not verify_password(password or "", u["password_hash"] if u else _DUMMY_HASH) or not u:
        log_event("login_failed", u["id"] if u else None, email, ip)
        raise AuthError("Wrong email or password", 401)
    db.update("users", u["id"], last_login_at=time.time())
    log_event("login", u["id"], email, ip)
    return u


def change_password(user: dict, current: str, new: str, keep_token: str | None, ip: str | None = None):
    u = _user(user["id"])
    if not verify_password(current or "", u["password_hash"]):
        log_event("password_change_failed", u["id"], u["email"], ip)
        raise AuthError("Your current password is wrong", 403)
    check_password(new, u["email"], u["name"])
    if verify_password(new, u["password_hash"]):
        raise AuthError("Choose a password you haven't used here")
    db.update("users", u["id"], password_hash=hash_password(new), password_changed_at=time.time())
    _end_sessions(u["id"], keep=keep_token)
    log_event("password_changed", u["id"], u["email"], ip)
    mailer.send(u["email"], "Your Eko Sahayak password was changed",
                f"Hi {u['name']},\n\nThe password for your Eko Sahayak account was just changed. If this wasn't you, "
                "reset it now from the sign-in page and tell your admin.\n")


# ---------------------------------------------------------------- one-time links (reset / verify)
def _new_link(user: dict, purpose: str, email: str, seconds: float) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.x("DELETE FROM auth_tokens WHERE (user_id=? AND purpose=? AND used_at IS NULL) OR expires_at<?",
         (user["id"], purpose, now - 86400))
    db.x("INSERT INTO auth_tokens(user_id,purpose,token_hash,email,created_at,expires_at) VALUES(?,?,?,?,?,?)",
         (user["id"], purpose, _h(token), email, now, now + seconds))
    return token


def _use_link(token: str, purpose: str) -> tuple[dict, dict]:
    row = db.q("SELECT * FROM auth_tokens WHERE token_hash=? AND purpose=?", (_h(token or ""), purpose), one=True)
    if not row or row["used_at"] or row["expires_at"] < time.time():
        raise AuthError("This link has expired or was already used. Ask for a new one.", 400)
    u = _user(row["user_id"])
    if not u:
        raise AuthError("This link has expired or was already used. Ask for a new one.", 400)
    db.x("UPDATE auth_tokens SET used_at=? WHERE id=?", (time.time(), row["id"]))
    return row, u


def request_password_reset(email: str, ip: str | None = None) -> bool:
    """Emails a reset link if the account exists. The caller always shows the same reply either way."""
    email = (email or "").strip().lower()[:254]
    u = db.q("SELECT * FROM users WHERE email=?", (email,), one=True)
    log_event("reset_requested", u["id"] if u else None, email, ip)
    if not u:
        return False
    # Counted from the log: each new link replaces the previous unused one, so the tokens table can't tell
    recent = db.q("SELECT COUNT(*) n FROM auth_events WHERE user_id=? AND event='reset_sent' AND ts>?",
                  (u["id"], time.time() - 3600), one=True)["n"]
    if recent >= MAX_RESETS_PER_HOUR or not mailer.configured():
        return False
    token = _new_link(u, "reset", u["email"], RESET_MINUTES * 60)
    log_event("reset_sent", u["id"], u["email"], ip)
    return mailer.send(u["email"], "Reset your Eko Sahayak password",
                       f"Hi {u['name']},\n\nSomeone (hopefully you) asked to reset your Eko Sahayak password. "
                       f"Open this link within {RESET_MINUTES} minutes to choose a new one:\n\n"
                       f"{mailer.link('reset', token)}\n\nIf you didn't ask, ignore this email; your password "
                       "stays the same.\n")


def admin_reset_link(actor: dict, user_id: int) -> str:
    """For when email isn't set up: the admin passes this one-time link to the user directly."""
    u = _user(user_id)
    if not u:
        raise AuthError("User not found", 404)
    log_event("reset_link_made", u["id"], u["email"], detail=f"by {actor['email']}")
    return _new_link(u, "reset", u["email"], RESET_MINUTES * 60)


def reset_password(token: str, new: str, ip: str | None = None) -> dict:
    row = db.q("SELECT user_id FROM auth_tokens WHERE token_hash=? AND purpose='reset'", (_h(token or ""),), one=True)
    target = _user(row["user_id"]) if row else None
    if target:
        check_password(new, target["email"], target["name"])   # before the link is used up
    row, u = _use_link(token, "reset")
    fields = {"password_hash": hash_password(new), "password_changed_at": time.time()}
    if row["email"] == u["email"] and not u.get("email_verified_at"):
        fields["email_verified_at"] = time.time()   # the link reached their inbox, so the address is theirs
    db.update("users", u["id"], **fields)
    _end_sessions(u["id"])
    log_event("password_reset", u["id"], u["email"], ip)
    if fields.get("email_verified_at"):
        _link_rm_by_email(_user(u["id"]))
    return _user(u["id"])


def send_verification(u: dict, new_email: str | None = None) -> bool:
    """Emails a link that proves the address is theirs. With new_email, the email changes only once it's clicked."""
    if not mailer.configured():
        return False
    target = new_email or u["email"]
    token = _new_link(u, "verify", target, VERIFY_HOURS * 3600)
    what = "change your Eko Sahayak email to this address" if new_email else "confirm your Eko Sahayak email"
    return mailer.send(target, "Confirm your email for Eko Sahayak",
                       f"Hi {u['name']},\n\nOpen this link to {what}:\n\n{mailer.link('verify-email', token)}\n\n"
                       f"It works for {VERIFY_HOURS} hours. If you didn't ask for this, ignore this email.\n")


def start_email_change(user: dict, new_email: str, password: str, ip: str | None = None) -> bool:
    u = _user(user["id"])
    if not verify_password(password or "", u["password_hash"]):
        raise AuthError("Your password is wrong", 403)
    new_email = _clean_email(new_email)
    if new_email == u["email"]:
        raise AuthError("That's already your email")
    if db.q("SELECT id FROM users WHERE email=?", (new_email,), one=True):
        raise AuthError("Another account uses that email", 409)
    if not mailer.configured():
        raise AuthError("Email isn't set up on this server yet; ask an admin to change it", 503)
    log_event("email_change_requested", u["id"], u["email"], ip, detail=new_email)
    mailer.send(u["email"], "Your Eko Sahayak email is being changed",
                f"Hi {u['name']},\n\nSomeone asked to move your Eko Sahayak account to {new_email}. It changes only "
                "when the link sent there is opened. If this wasn't you, change your password now.\n")
    return send_verification(u, new_email)


def verify_email(token: str, ip: str | None = None) -> dict:
    row, u = _use_link(token, "verify")
    if row["email"] != u["email"]:
        if db.q("SELECT id FROM users WHERE email=? AND id<>?", (row["email"], u["id"]), one=True):
            raise AuthError("Another account uses that email now", 409)
        log_event("email_changed", u["id"], row["email"], ip, detail=f"was {u['email']}")
        db.update("users", u["id"], email=row["email"])
    db.update("users", u["id"], email_verified_at=time.time())
    log_event("email_verified", u["id"], row["email"], ip)
    return _link_rm_by_email(_user(u["id"]))


def _link_rm_by_email(u: dict) -> dict:
    """A verified email that the calling sheet gives for an RM links the account to that RM profile."""
    if u.get("rm_id") or not u.get("email_verified_at"):
        return u
    rm = db.q("""SELECT id, name FROM rms WHERE lower(email)=? AND id NOT IN
                 (SELECT rm_id FROM users WHERE rm_id IS NOT NULL)""", (u["email"].lower(),), one=True)
    if rm:
        db.update("users", u["id"], rm_id=rm["id"])
        log_event("rm_linked", u["id"], u["email"], detail=f"RM {rm['name']} by verified email")
    return _user(u["id"])


# ---------------------------------------------------------------- sessions
def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int, ip: str | None = None, user_agent: str | None = None) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.x("DELETE FROM sessions WHERE expires_at < ? OR last_seen_at < ?", (now, now - SESSION_IDLE_HOURS * 3600))
    db.x("INSERT INTO sessions(token_hash,user_id,created_at,expires_at,last_seen_at,ip,user_agent)"
         " VALUES(?,?,?,?,?,?,?)", (_h(token), user_id, now, now + SESSION_DAYS * 86400, now, ip,
                                    (user_agent or "")[:200]))
    return token


def user_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    now = time.time()
    s = db.q("SELECT last_seen_at FROM sessions WHERE token_hash=? AND expires_at>?", (_h(token), now), one=True)
    if not s:
        return None
    if s["last_seen_at"] and s["last_seen_at"] < now - SESSION_IDLE_HOURS * 3600:
        end_session(token)
        return None
    if not s["last_seen_at"] or now - s["last_seen_at"] > 300:   # write at most every 5 minutes
        db.x("UPDATE sessions SET last_seen_at=? WHERE token_hash=?", (now, _h(token)))
    return db.q("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?", (_h(token),),
                one=True)


def end_session(token: str | None):
    if token:
        db.x("DELETE FROM sessions WHERE token_hash=?", (_h(token),))


def _end_sessions(user_id: int, keep: str | None = None):
    db.x("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (user_id, _h(keep) if keep else ""))


def end_all_sessions(user: dict, ip: str | None = None):
    _end_sessions(user["id"])
    log_event("signed_out_everywhere", user["id"], user["email"], ip)


def session_count(user_id: int) -> int:
    return db.q("SELECT COUNT(*) n FROM sessions WHERE user_id=? AND expires_at>?", (user_id, time.time()),
                one=True)["n"]
