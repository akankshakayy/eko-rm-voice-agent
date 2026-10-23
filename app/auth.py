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
# "pending" is no longer given at signup; an admin can still set it to take someone's access away
ROLE_LABELS = {"admin": "Admin", "ops": "Ops", "rm": "RM", "pending": "No access"}
# New accounts start with this role, or RM when their verified email is an RM's on the calling sheet
DEFAULT_ROLE = os.environ.get("DEFAULT_ROLE", "ops") if os.environ.get("DEFAULT_ROLE", "ops") in ("ops", "pending") else "ops"
def allow_unverified() -> bool:
    """Local testing only: password signups work without email, unconfirmed. Never on a shared server."""
    return os.environ.get("ALLOW_UNVERIFIED_SIGNUP", "0") == "1"
IP_FAILS, IP_FAIL_WINDOW = 20, 15 * 60          # failed sign-ins from one address before it has to wait
IP_REQUESTS, IP_REQUEST_WINDOW = 10, 3600       # signups / reset emails from one address per hour

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


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:                      # an account made with Google has no password
        return False
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


def admin_emails() -> set[str]:
    """ADMIN_EMAILS: who becomes admin. Unset, the very first account does (fine on a laptop, not on a fresh server
    anyone with a company address could reach first)."""
    return {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}


def _new_role(email: str) -> str:
    admins = admin_emails()
    if admins:
        return "admin" if email in admins else DEFAULT_ROLE
    return "admin" if not db.q("SELECT id FROM users LIMIT 1", one=True) else DEFAULT_ROLE


def password_signup_open() -> bool:
    """A password signup must prove the email is theirs by a confirmation email, so it needs email set up.
    (The very first account, the admin who sets the server up, is the exception.)"""
    return mailer.configured() or allow_unverified() or not db.q("SELECT id FROM users LIMIT 1", one=True)


def _ip_busy(ip: str | None, events: tuple, limit: int, window: float) -> bool:
    if not ip:
        return False
    marks = ",".join("?" * len(events))
    n = db.q(f"SELECT COUNT(*) n FROM auth_events WHERE ip=? AND event IN ({marks}) AND ts>?",
             (ip, *events, time.time() - window), one=True)["n"]
    return n >= limit


def signup(email: str, password: str, name: str, rm_id: int | None = None, ip: str | None = None) -> dict:
    """A working account straight away, no admin approval: but a password account can be used only once its
    email is confirmed (the link in the email), because a company address alone proves nothing.
    rm_id is ignored: the RM profile is linked by a verified email, or by an admin."""
    if _ip_busy(ip, ("signup",), IP_REQUESTS, IP_REQUEST_WINDOW):
        raise AuthError("Too many new accounts from here. Try again in an hour.", 429)
    email = _clean_email(email)
    name = (name or "").strip()
    if not name or len(name) > 100:
        raise AuthError("Enter your name")
    if not password_signup_open():
        raise AuthError("Creating an account with a password needs email on this server, to confirm the address "
                        "is yours. Use Sign in with Google, or ask an admin.", 403)
    check_password(password, email, name)
    if db.q("SELECT id FROM users WHERE email=?", (email,), one=True):
        raise AuthError("An account with this email already exists. Sign in, or reset the password.", 409)
    first = not db.q("SELECT id FROM users LIMIT 1", one=True)
    must_verify = 0 if first or allow_unverified() else 1
    now = time.time()
    uid = db.x("INSERT INTO users(email,name,password_hash,role,created_at,password_changed_at,must_verify)"
               " VALUES(?,?,?,?,?,?,?)", (email, name, hash_password(password), _new_role(email), now, now,
                                          must_verify))
    log_event("signup", uid, email, ip)
    u = _user(uid)
    if must_verify:
        send_verification(u)
    return u


def google_sign_in(email: str, name: str, sub: str, ip: str | None = None) -> dict:
    """Google has verified this address: find the account (by Google id, then email) or make one, ready to use."""
    email = _clean_email(email)                       # company domains only, whatever Google says
    now = time.time()
    u = db.q("SELECT * FROM users WHERE google_sub=?", (sub,), one=True) or \
        db.q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if u:
        fields = {"last_login_at": now, "must_verify": 0}
        if not u.get("google_sub"):
            fields["google_sub"] = sub
        if not u.get("email_verified_at"):
            # Nobody had proven this address until now, so a password set on it may be someone else's (an account
            # prepared in advance to take over the real owner's): drop that password and its sessions
            fields.update(email_verified_at=now, password_hash=None)
            _end_sessions(u["id"])
            log_event("unverified_password_dropped", u["id"], email, ip, detail="first Google sign-in")
        db.update("users", u["id"], **fields)
        log_event("login", u["id"], email, ip, detail="google")
        return _link_rm_by_email(_user(u["id"]))
    uid = db.x("INSERT INTO users(email,name,role,created_at,last_login_at,email_verified_at,google_sub,must_verify)"
               " VALUES(?,?,?,?,?,?,?,0)", (email, (name or email.split("@")[0])[:100], _new_role(email), now, now,
                                            now, sub))
    log_event("signup", uid, email, ip, detail="google")
    return _link_rm_by_email(_user(uid))


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
    if _locked(email) or _ip_busy(ip, ("login_failed",), IP_FAILS, IP_FAIL_WINDOW):
        log_event("login_locked", email=email, ip=ip)
        raise AuthError("Too many failed attempts. Try again in 15 minutes, or reset your password.", 429)
    u = db.q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if not verify_password(password or "", (u["password_hash"] or _DUMMY_HASH) if u else _DUMMY_HASH) or not u:
        log_event("login_failed", u["id"] if u else None, email, ip)
        hint = " This account uses Sign in with Google." if u and not u["password_hash"] else ""
        raise AuthError("Wrong email or password." + hint, 401)
    if u.get("must_verify") and not u.get("email_verified_at"):
        # Only after the right password, so this can't be used to find out who has an account
        raise AuthError(f"Confirm your email first: open the link we sent to {u['email']}.", 403)
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
    if recent >= MAX_RESETS_PER_HOUR or not mailer.configured() or \
            _ip_busy(ip, ("reset_sent",), IP_REQUESTS, IP_REQUEST_WINDOW):
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


def resend_confirmation(email: str, password: str, ip: str | None = None) -> bool:
    """For someone who signed up but lost the confirmation email: the right password sends it again."""
    if _ip_busy(ip, ("confirmation_resent",), IP_REQUESTS, IP_REQUEST_WINDOW):
        raise AuthError("Too many requests from here. Try again in an hour.", 429)
    u = db.q("SELECT * FROM users WHERE email=?", ((email or "").strip().lower()[:254],), one=True)
    if not u or not verify_password(password or "", u["password_hash"]):
        raise AuthError("Wrong email or password.", 401)
    if u.get("email_verified_at"):
        return False
    log_event("confirmation_resent", u["id"], u["email"], ip)
    return send_verification(u)


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
    db.update("users", u["id"], email_verified_at=time.time(), must_verify=0)
    log_event("email_verified", u["id"], row["email"], ip)
    return _link_rm_by_email(_user(u["id"]))


def _link_rm_by_email(u: dict) -> dict:
    """A verified email that the calling sheet gives for an RM links the account to that RM profile."""
    if u.get("rm_id") or not u.get("email_verified_at"):
        return u
    rm = db.q("""SELECT id, name FROM rms WHERE lower(email)=? AND id NOT IN
                 (SELECT rm_id FROM users WHERE rm_id IS NOT NULL)""", (u["email"].lower(),), one=True)
    if rm:
        # An RM's own email: they get the RM role for their CSPs, unless an admin gave them more already
        db.update("users", u["id"], rm_id=rm["id"], role="rm" if u.get("role") == "ops" else u["role"])
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
