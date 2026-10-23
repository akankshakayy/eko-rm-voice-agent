"""FastAPI app: agent API + dashboards.

Run:  uvicorn app.main:app --reload --port 8000
"""
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import quote, urlparse

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import agent, auth, db, google_auth, integrations, mailer, metrics, recordings, report, stt, tts  # noqa: E402
from .seed import seed_if_demo  # noqa: E402
from .taxonomy import CATEGORIES, SLOT_LABELS, STT_CONFIDENCE_MIN, INTENT_CONFIDENCE_MIN, MAX_MISUNDERSTOOD  # noqa

STATIC = os.path.join(os.path.dirname(__file__), "..", "static")
log = logging.getLogger("eko")


def _reap_stale_calls(stop: threading.Event):
    """Every minute, close calls nobody has touched for CALL_IDLE_TIMEOUT_MIN minutes."""
    while not stop.is_set():
        try:
            closed = agent.close_stale_calls()
            if closed:
                log.warning("Closed %d idle call(s): %s", len(closed), closed)
        except Exception:
            log.exception("Idle-call cleanup failed")
        stop.wait(60)


@asynccontextmanager
async def lifespan(_app):
    stop = threading.Event()
    threading.Thread(target=_reap_stale_calls, args=(stop,), daemon=True, name="call-reaper").start()
    threading.Thread(target=recordings.processing_loop, args=(stop,), daemon=True, name="recordings").start()
    threading.Thread(target=integrations.delivery_loop, args=(stop,), daemon=True, name="webhooks").start()
    yield
    stop.set()
    integrations.wake()
    recordings.wake()


app = FastAPI(title="Eko RM Voice Agent", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
seed_if_demo()

# ------------------------------------------------------------------ auth
PUBLIC_PATHS = {"/login", "/signup", "/forgot", "/reset", "/verify-email", "/favicon.ico", "/api/auth/login",
                "/api/auth/signup", "/api/auth/forgot", "/api/auth/reset", "/api/auth/verify", "/api/auth/options",
                "/api/auth/verify/send", "/api/auth/google/start", "/api/auth/google/callback"}
PUBLIC_PREFIXES = ("/static/",)
DEVICE_PATHS = {"/api/uploader/ping", "/api/uploader/recordings"}   # the uploader app: upload token, not a login
STAFF = ("admin", "ops")                       # full access to calls, dashboards and metrics
PAGE_ROLES = {"/": ("admin", "ops", "rm"), "/call": STAFF, "/ops": STAFF, "/rm": ("admin", "ops", "rm"),
              "/users": ("admin",), "/recordings": ("admin", "ops", "rm"), "/account": ("admin", "ops", "rm", "pending")}
# All a not-yet-approved account can reach: its own account settings
PENDING_OK = {"/pending", "/account", "/api/auth/me", "/api/auth/logout", "/api/auth/logout-all", "/api/auth/password",
              "/api/auth/email", "/api/auth/verify/resend", "/api/auth/activity"}
TRUST_PROXY = os.environ.get("TRUST_PROXY", "0") == "1"   # behind nginx: take the client address from X-Forwarded-For
# The login cookie is sent only to Sahayak's own path. On a shared host (indev.eko.in runs other people's apps too)
# a cookie for "/" would reach every app there. Taken from APP_BASE_URL (".../rm-voice-agent"), or COOKIE_PATH.
def cookie_path() -> str:
    return (os.environ.get("COOKIE_PATH") or urlparse(os.environ.get("APP_BASE_URL", "")).path or "/").rstrip("/") or "/"


def client_ip(request: Request) -> str | None:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if fwd:
            return fwd[:64]
    return request.client.host if request.client else None




# Redirects and page links are relative ("login", "static/..."), so Sahayak works at the top of an address
# (localhost:8000) and under a prefix (https://indev.eko.in/rm-voice-agent) alike. Every page is one level deep.
@app.middleware("http")
async def require_login(request: Request, call_next):
    """Every page and API needs a signed-in user, except the login/signup pages and static files.
    Pages are also checked against the user's role here; API routes check roles with `allow(...)`."""
    path = request.url.path
    if path in PUBLIC_PATHS or path in DEVICE_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and (path.startswith("/api/") or path == "/openapi.json"):
        # Another system using a service API key (app/integrations.py): acts as Ops, no cookie involved
        service = integrations.service_for_key(header[7:].strip())
        if not service:
            return JSONResponse({"detail": "API key is wrong or revoked"}, status_code=401)
        request.state.user = service
        return await call_next(request)
    user = auth.user_for_token(request.cookies.get(auth.COOKIE))
    if not user:
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Not signed in"}, status_code=401)
        nxt = path.lstrip("/") if path in PAGE_ROLES and path != "/" else "./"
        return RedirectResponse(f"login?next={nxt}", status_code=303)
    role = user.get("role") or "pending"
    if role == "pending" and path not in PENDING_OK:
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Your account has no access. Ask an admin."}, status_code=403)
        return RedirectResponse("pending", status_code=303)
    if role != "pending" and path == "/pending":
        return RedirectResponse("./", status_code=303)
    if path in PAGE_ROLES and role not in PAGE_ROLES[path]:
        return RedirectResponse("./?denied=1", status_code=303)
    request.state.user = user
    return await call_next(request)


# Added after require_login, so it runs first and also covers its redirects and 401s
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    # Reset and verification links carry a token in the address; never pass it on to another site
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Permissions-Policy", "camera=(), geolocation=()")
    if request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    if auth.COOKIE_SECURE:
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    return resp


def allow(*roles):
    """Route dependency: only these roles may call the endpoint."""
    def check(request: Request):
        if request.state.user.get("role") not in roles:
            raise HTTPException(403, "You don't have access to this")
        return request.state.user
    return check


def rm_scope(request: Request, rm_id: int | None) -> int | None:
    """RMs only ever see their own data, whatever rm_id they ask for."""
    u = request.state.user
    return u["rm_id"] if u.get("role") == "rm" else rm_id


def check_rm_owns(request: Request, rm_id: int | None, what: str):
    u = request.state.user
    if u.get("role") == "rm" and rm_id != u["rm_id"]:
        raise HTTPException(404, f"{what} not found")   # 404, not 403: don't reveal other RMs' data exists


class SignupBody(BaseModel):
    email: str
    password: str
    name: str
    rm_id: int | None = None


class LoginBody(BaseModel):
    email: str
    password: str


class EmailBody(BaseModel):
    email: str


class ResetBody(BaseModel):
    token: str
    password: str


class TokenBody(BaseModel):
    token: str


class PasswordBody(BaseModel):
    current: str
    new: str


class EmailChangeBody(BaseModel):
    email: str
    password: str


def _signed_in(user: dict, request: Request) -> JSONResponse:
    resp = JSONResponse(auth.public_user(user))
    token = auth.create_session(user["id"], client_ip(request), request.headers.get("user-agent"))
    _set_session_cookie(resp, token)
    return resp


def _set_session_cookie(resp, token: str):
    resp.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400,
                    httponly=True, samesite="lax", secure=auth.COOKIE_SECURE, path=cookie_path())


def _clear_session_cookie(resp):
    resp.delete_cookie(auth.COOKIE, path=cookie_path())
    if cookie_path() != "/":
        resp.delete_cookie(auth.COOKIE, path="/")          # one set by an older version, for the whole host


def _auth_call(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e))


@app.post("/api/auth/signup")
def signup(body: SignupBody, request: Request):
    u = _auth_call(auth.signup, body.email, body.password, body.name, body.rm_id, client_ip(request))
    if u.get("must_verify"):
        # Not signed in yet: the account works once the link in the email is opened
        return {"verify": True, "email": u["email"]}
    return _signed_in(u, request)


@app.post("/api/auth/verify/send")
def send_confirmation_again(body: LoginBody, request: Request):
    return {"sent": _auth_call(auth.resend_confirmation, body.email, body.password, client_ip(request))}


@app.get("/api/auth/google/start")
def google_start(next: str = "./"):
    """Off to Google's sign-in page; it comes back to /api/auth/google/callback."""
    if not google_auth.configured():
        return RedirectResponse("../../../login?error=" + quote("Sign in with Google isn't set up on this server."), 303)
    url, flow = google_auth.start(next if re.fullmatch(r"(\./|[a-z][a-z-]*|/(?!/)[\w/.-]*)", next or "") else "./",
                                  auth.allowed_domains())
    resp = RedirectResponse(url, 303)
    resp.set_cookie(google_auth.COOKIE, flow, max_age=google_auth.FLOW_SECONDS, httponly=True, samesite="lax",
                    secure=auth.COOKIE_SECURE, path=cookie_path())
    return resp


@app.get("/api/auth/google/callback")
def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    back = "../../../"                                  # from /api/auth/google/callback to the site's root
    try:
        if error:
            raise google_auth.GoogleError("Google sign-in was cancelled." if error == "access_denied" else
                                          f"Google sign-in failed ({error[:40]}).")
        who = google_auth.finish(code, state, request.cookies.get(google_auth.COOKIE))
        u = auth.google_sign_in(who["email"], who["name"], who["sub"], client_ip(request))
    except (google_auth.GoogleError, auth.AuthError) as e:
        resp = RedirectResponse(back + "login?error=" + quote(str(e)), 303)
    else:
        nxt = who["next"]                               # checked at start: a page name, ./ or a same-site path
        resp = RedirectResponse(nxt if nxt.startswith("/") else back + nxt, 303)
        _set_session_cookie(resp, auth.create_session(u["id"], client_ip(request), request.headers.get("user-agent")))
    resp.delete_cookie(google_auth.COOKIE, path=cookie_path())
    return resp


@app.post("/api/auth/login")
def login(body: LoginBody, request: Request):
    return _signed_in(_auth_call(auth.login, body.email, body.password, client_ip(request)), request)


@app.get("/api/auth/options")
def auth_options():
    """What the sign-in pages can offer: email links only work once SMTP is set up."""
    return {"email": mailer.configured(), "min_password": auth.MIN_PASSWORD, "domains": auth.allowed_domains(),
            "google": google_auth.configured(), "password_signup": auth.password_signup_open()}


@app.post("/api/auth/forgot")
def forgot(body: EmailBody, request: Request):
    auth.request_password_reset(body.email, client_ip(request))
    # The same reply whether or not the account exists, so this can't be used to find out who has one
    return {"ok": True}


@app.post("/api/auth/reset")
def reset(body: ResetBody, request: Request):
    _auth_call(auth.reset_password, body.token, body.password, client_ip(request))
    resp = JSONResponse({"ok": True})
    _clear_session_cookie(resp)
    return resp


@app.post("/api/auth/verify")
def verify(body: TokenBody, request: Request):
    return auth.public_user(_auth_call(auth.verify_email, body.token, client_ip(request)))


@app.post("/api/auth/verify/resend")
def resend_verification(request: Request):
    u = request.state.user
    if u.get("email_verified_at"):
        return {"sent": False, "verified": True}
    return {"sent": auth.send_verification(u)}


@app.post("/api/auth/password")
def change_password(body: PasswordBody, request: Request):
    _auth_call(auth.change_password, request.state.user, body.current, body.new, request.cookies.get(auth.COOKIE),
               client_ip(request))
    return {"ok": True}


@app.post("/api/auth/email")
def change_email(body: EmailChangeBody, request: Request):
    return {"sent": _auth_call(auth.start_email_change, request.state.user, body.email, body.password,
                               client_ip(request))}


@app.post("/api/auth/logout-all")
def logout_all(request: Request):
    auth.end_all_sessions(request.state.user, client_ip(request))
    resp = JSONResponse({"ok": True})
    _clear_session_cookie(resp)
    return resp


@app.get("/api/auth/activity")
def my_activity(request: Request):
    u = request.state.user
    return {"events": auth.activity(u["id"]), "sessions": auth.session_count(u["id"]),
            "password_changed_at": u.get("password_changed_at")}


@app.post("/api/auth/logout")
def logout(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    resp = JSONResponse({"ok": True})
    _clear_session_cookie(resp)
    return resp


@app.get("/api/auth/me")
def me(request: Request):
    return auth.public_user(request.state.user)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(STATIC, "favicon.ico"), media_type="image/x-icon")


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


@app.get("/signup")
@app.get("/forgot")
@app.get("/reset")
@app.get("/verify-email")
def signup_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


@app.get("/account")
def account_page():
    return FileResponse(os.path.join(STATIC, "account.html"))


@app.get("/pending")
def pending_page():
    return FileResponse(os.path.join(STATIC, "pending.html"))


@app.get("/users")
def users_page():
    return FileResponse(os.path.join(STATIC, "users.html"))


# ------------------------------------------------------------------ user management (admin)
class RoleBody(BaseModel):
    role: str
    rm_id: int | None = None


@app.get("/api/users", dependencies=[Depends(allow("admin"))])
def list_users():
    return [auth.public_user(u) for u in db.q("SELECT * FROM users ORDER BY role='pending' DESC, created_at")]


@app.patch("/api/users/{user_id}")
def change_role(user_id: int, body: RoleBody, actor=Depends(allow("admin"))):
    try:
        return auth.public_user(auth.set_role(actor, user_id, body.role, body.rm_id))
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e))


@app.post("/api/users/{user_id}/reset-link")
def user_reset_link(user_id: int, actor=Depends(allow("admin"))):
    """One-time password reset link for a user, for when email isn't set up. Shown to the admin only once."""
    return {"path": f"reset?token={_auth_call(auth.admin_reset_link, actor, user_id)}",
            "minutes": auth.RESET_MINUTES}


@app.delete("/api/users/{user_id}")
def remove_user(user_id: int, actor=Depends(allow("admin"))):
    try:
        auth.delete_user(actor, user_id)
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/call")
def call_page():
    return FileResponse(os.path.join(STATIC, "call.html"))


@app.get("/rm")
def rm_page():
    return FileResponse(os.path.join(STATIC, "rm.html"))


@app.get("/ops")
def ops_page():
    return FileResponse(os.path.join(STATIC, "ops.html"))


@app.get("/recordings")
def recordings_page():
    return FileResponse(os.path.join(STATIC, "recordings.html"))


# ------------------------------------------------------------------ reference data
@app.get("/api/config", dependencies=[Depends(allow("admin", "ops", "rm"))])
def config():
    from .llm_engine import label_for, model_for
    mode = agent.engine_mode()
    return {"engine": mode, "engine_label": label_for(mode), "model": model_for(mode), "stt": stt.available(),
            "thresholds": {"stt_confidence_min": STT_CONFIDENCE_MIN, "intent_confidence_min": INTENT_CONFIDENCE_MIN,
                           "max_misunderstood": MAX_MISUNDERSTOOD},
            "categories": {k: {"label": v["label"], "priority": v["priority"], "slots": v["slots"]}
                           for k, v in CATEGORIES.items()},
            "slot_labels": SLOT_LABELS}


@app.get("/api/rms", dependencies=[Depends(allow("admin", "ops", "rm"))])
def rms():
    return db.q("SELECT * FROM rms ORDER BY id")


@app.get("/api/csps", dependencies=[Depends(allow(*STAFF))])
def csps(rm_id: int | None = None):
    if rm_id:
        return db.q("SELECT * FROM csps WHERE rm_id=? ORDER BY name", (rm_id,))
    return db.q("SELECT c.*, r.name rm_name FROM csps c JOIN rms r ON r.id=c.rm_id ORDER BY c.rm_id, c.name")


@app.get("/api/console/csps", dependencies=[Depends(allow(*STAFF))])
def console_csps(rm_id: int | None = None, q: str = "", limit: int = 40):
    """The call console's CSP picker for one RM (or all): test CSPs, which can be called, and real CSPs from the
    knowledge base, which are view only while real calls are off. Search matches the CSP code or name."""
    where, args = ["c.source IN ({})"], []
    if rm_id:
        where.append("c.rm_id=?")
        args.append(rm_id)
    q = q.strip()[:60]
    if q:
        like = "%" + q.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append("(lower(c.csp_code) LIKE ? ESCAPE '\\' OR lower(c.name) LIKE ? ESCAPE '\\')")
        args += [like, like]
    cond = " AND ".join(where)
    limit = max(1, min(limit, 200))
    month = time.strftime("%Y-%m")

    def part(sources):
        marks = ",".join("?" * len(sources))
        rows = db.q(f"""SELECT c.id, c.csp_code, c.name, c.shop_name, c.district, c.state, c.category, c.active,
                               c.source, c.rm_id, r.name rm_name, m.slab
                        FROM csps c LEFT JOIN rms r ON r.id=c.rm_id
                        LEFT JOIN csp_monthly m ON m.csp_id=c.id AND m.month=?
                        WHERE {cond.format(marks)} ORDER BY c.active DESC, c.name LIMIT ?""",
                    (month, *sources, *args, limit))
        total = db.q(f"SELECT COUNT(*) n FROM csps c WHERE {cond.format(marks)}", (*sources, *args), one=True)["n"]
        return rows, total

    test, n_test = part(("synthetic", "demo"))
    real, n_real = part(("kb",))
    return {"test": test, "test_total": n_test, "real": real, "real_total": n_real,
            "real_calls": agent.real_calls_on()}


# ------------------------------------------------------------------ calls
class StartCall(BaseModel):
    direction: str = "inbound"
    phone: str | None = None
    purpose: str = "support"
    lang: str = "hi"   # hi | en | ask (agent asks the caller to pick)
    csp_id: int | None = None   # the caller picked from the console's CSP list


class Turn(BaseModel):
    text: str
    stt_confidence: float | None = None


@app.post("/api/calls", dependencies=[Depends(allow(*STAFF))])
def start_call(body: StartCall):
    try:
        return agent.start_call(body.direction, body.phone, body.purpose, lang=body.lang, csp_id=body.csp_id)
    except agent.RealCspBlocked as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/calls/{call_id}/turn", dependencies=[Depends(allow(*STAFF))])
def turn(call_id: int, body: Turn):
    try:
        return agent.handle_turn(call_id, body.text, body.stt_confidence)
    except KeyError:
        raise HTTPException(404, "call not found")


@app.post("/api/calls/{call_id}/hangup", dependencies=[Depends(allow(*STAFF))])
def hangup(call_id: int):
    try:
        return agent.hangup(call_id)
    except KeyError:
        raise HTTPException(404, "call not found")


@app.get("/api/calls/{call_id}", dependencies=[Depends(allow("admin", "ops", "rm"))])
def get_call(call_id: int, request: Request):
    try:
        c = agent.public_call(call_id)
    except Exception:
        raise HTTPException(404, "call not found")
    check_rm_owns(request, c["rm_id"], "call")
    c["turns"] = db.q("SELECT role,text,ts,stt_confidence,latency_ms,engine,intent,intent_confidence,error "
                      "FROM turns WHERE call_id=? ORDER BY id", (call_id,))
    return c


# ------------------------------------------------------------------ tickets
@app.get("/api/tickets", dependencies=[Depends(allow("admin", "ops", "rm"))])
def tickets(request: Request, rm_id: int | None = None, status: str | None = None, escalated: int | None = None,
            limit: int = 200):
    rm_id = rm_scope(request, rm_id)
    where, args = ["1=1"], []
    if rm_id:
        where.append("t.rm_id=?"); args.append(rm_id)
    if status:
        where.append("t.status IN (%s)" % ",".join("?" * len(status.split(",")))); args += status.split(",")
    if escalated is not None:
        where.append("t.escalated=?"); args.append(escalated)
    rows = db.q(f"""SELECT t.*, s.name csp_name, s.csp_code, s.shop_name, s.phone csp_phone, s.district,
                    c.duration_sec, c.sentiment, c.direction
                    FROM tickets t LEFT JOIN csps s ON s.id=t.csp_id LEFT JOIN calls c ON c.id=t.call_id
                    WHERE {' AND '.join(where)} ORDER BY t.created_at DESC LIMIT ?""", (*args, limit))
    for r in rows:
        r["details"] = json.loads(r.pop("details_json") or "{}")
    return rows


@app.get("/api/tickets/{ticket_id}", dependencies=[Depends(allow("admin", "ops", "rm"))])
def ticket(ticket_id: int, request: Request):
    t = db.q("""SELECT t.*, s.name csp_name, s.csp_code, s.shop_name, s.phone csp_phone, s.district, r.name rm_name
                FROM tickets t LEFT JOIN csps s ON s.id=t.csp_id LEFT JOIN rms r ON r.id=t.rm_id WHERE t.id=?""",
             (ticket_id,), one=True)
    if not t:
        raise HTTPException(404, "ticket not found")
    check_rm_owns(request, t["rm_id"], "ticket")
    t["details"] = json.loads(t.pop("details_json") or "{}")
    t["events"] = db.q("SELECT * FROM ticket_events WHERE ticket_id=? ORDER BY ts", (ticket_id,))
    t["transcript"] = db.q("SELECT role,text,ts,stt_confidence FROM turns WHERE call_id=? ORDER BY id", (t["call_id"],))
    return t


class TicketUpdate(BaseModel):
    status: str | None = None
    rm_category: str | None = None
    note: str | None = None


@app.patch("/api/tickets/{ticket_id}", dependencies=[Depends(allow("admin", "ops", "rm"))])
def patch_ticket(ticket_id: int, body: TicketUpdate, request: Request):
    if body.status and body.status not in ("open", "in_progress", "resolved", "closed"):
        raise HTTPException(400, "bad status")
    owner = db.q("SELECT rm_id FROM tickets WHERE id=?", (ticket_id,), one=True)
    if not owner:
        raise HTTPException(404, "ticket not found")
    check_rm_owns(request, owner["rm_id"], "ticket")
    try:
        actor = request.state.user["name"]   # audit trail shows who changed the ticket
        return agent.update_ticket(ticket_id, actor, body.status, body.rm_category, body.note)
    except KeyError:
        raise HTTPException(404, "ticket not found")


# ------------------------------------------------------------------ metrics
@app.get("/api/metrics/agent", dependencies=[Depends(allow("admin", "ops", "rm"))])
def m_agent(request: Request, hours: float = 168, rm_id: int | None = None):
    return metrics.agent_metrics(hours, rm_scope(request, rm_id))


@app.get("/api/metrics/model", dependencies=[Depends(allow(*STAFF))])
def m_model(hours: float = 168):
    return metrics.model_metrics(hours)


# ------------------------------------------------------------------ voice
class Speak(BaseModel):
    text: str
    lang: str = "hi-IN"


@app.post("/api/tts", dependencies=[Depends(allow(*STAFF))])
async def speak(body: Speak):
    text = body.text.strip()[:1000]
    if not text:
        raise HTTPException(400, "empty text")
    try:
        audio = await tts.synthesize(text, body.lang)
    except Exception as e:
        raise HTTPException(502, f"tts failed: {e}")
    return Response(audio, media_type="audio/mpeg")


@app.post("/api/stt", dependencies=[Depends(allow(*STAFF))])
def transcribe(audio: UploadFile = File(...), lang: str = Form("hi-IN")):
    if not stt.available():
        raise HTTPException(503, "Server speech-to-text needs GROQ_API_KEY")
    data = audio.file.read()
    if not data:
        raise HTTPException(400, "empty audio")
    try:
        return {"text": stt.transcribe(data, audio.filename or "speech.webm", lang)}
    except Exception as e:
        raise HTTPException(502, f"stt failed: {e}")


# ------------------------------------------------------------------ call recordings (pilot, app/recordings.py)
def _device(request: Request) -> dict:
    """The uploader app signs in with its upload token: `Authorization: Bearer ekoup_...`."""
    header = request.headers.get("authorization", "")
    dev = recordings.device_for_token(header[7:].strip() if header.lower().startswith("bearer ") else None)
    if not dev:
        raise HTTPException(401, "Upload token is missing, wrong or revoked")
    return dev


@app.get("/api/uploader/ping")
def uploader_ping(device=Depends(_device)):
    """The app's setup screen: whose phone this is, or (shared token) the RMs to choose from. Only for a valid
    upload token, and only ids and names."""
    return {"ok": True, "rm_id": device["rm_id"], "rm_name": device["rm_name"], "rm_locked": bool(device["rm_id"]),
            "rms": recordings.rms_for_app()}


@app.post("/api/uploader/recordings")
def uploader_upload(audio: UploadFile = File(...), started_at: int | None = Form(None),
                    duration_sec: float | None = Form(None), direction: str = Form("unknown"),
                    other_number: str = Form(""), device_id: str = Form(""), rm_id: int | None = Form(None),
                    device=Depends(_device)):
    data = audio.file.read(recordings.MAX_BYTES + 1)
    if len(data) > recordings.MAX_BYTES:
        raise HTTPException(413, "Recording is over 100 MB")
    try:
        device = {**device, "rm_id": recordings.uploader_rm(device, rm_id)}
        return recordings.save_upload(device, data, audio.filename or "", started_at, duration_sec, direction,
                                      other_number, device_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


class UploadTokenBody(BaseModel):
    rm_id: int | None = None          # None: a shared token, each RM picks their name in the app
    label: str = ""


@app.get("/api/uploader/tokens", dependencies=[Depends(allow("admin"))])
def upload_tokens():
    return recordings.list_tokens()


@app.post("/api/uploader/tokens")
def new_upload_token(body: UploadTokenBody, actor=Depends(allow("admin"))):
    try:
        return recordings.create_token(body.rm_id, body.label, actor.get("name") or actor.get("email"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/uploader/tokens/{token_id}", dependencies=[Depends(allow("admin"))])
def revoke_upload_token(token_id: int):
    if not recordings.revoke_token(token_id):
        raise HTTPException(404, "No active token with that id")
    return {"ok": True}


@app.get("/api/recordings", dependencies=[Depends(allow("admin", "ops", "rm"))])
def list_recordings(request: Request, status: str | None = None, rm_id: int | None = None, limit: int = 500):
    rm_id = rm_scope(request, rm_id)
    return {"counts": recordings.summary_counts(rm_id), "recordings": recordings.list_recordings(status, rm_id, limit)}


def _own_recording(request: Request, rec_id: int) -> dict:
    r = recordings.get_recording(rec_id)
    if not r:
        raise HTTPException(404, "Recording not found")
    check_rm_owns(request, r["rm_id"], "Recording")
    return r


@app.get("/api/recordings/{rec_id}", dependencies=[Depends(allow("admin", "ops", "rm"))])
def get_recording(rec_id: int, request: Request):
    return _own_recording(request, rec_id)


@app.get("/api/recordings/{rec_id}/audio", dependencies=[Depends(allow("admin", "ops", "rm"))])
def recording_audio(rec_id: int, request: Request):
    _own_recording(request, rec_id)
    row = db.q("SELECT file_path FROM recordings WHERE id=?", (rec_id,), one=True)
    path = recordings.audio_file(row)
    if not os.path.exists(path):
        raise HTTPException(404, "The audio file is missing")
    return FileResponse(path)


@app.post("/api/recordings/upload")
def upload_recordings(audio: list[UploadFile] = File(...), rm_id: int | None = Form(None), started_at: str = Form(""),
                      direction: str = Form("unknown"), other_number: str = Form(""), actor=Depends(allow(*STAFF))):
    """Add recordings by hand (e.g. calls shared on WhatsApp). They go through the same processing as phone uploads.
    started_at: one epoch-ms value per file, comma-separated (the page sends each file's own date)."""
    rm_id = rm_id or None                     # 0 / empty = not an RM call (e.g. a meeting or an interview)
    if rm_id and not db.q("SELECT id FROM rms WHERE id=?", (rm_id,), one=True):
        raise HTTPException(400, "No such RM")
    times = [t.strip() for t in started_at.split(",")]
    results = []
    for i, f in enumerate(audio):
        data = f.file.read(recordings.MAX_BYTES + 1)
        name = f.filename or "recording"
        if len(data) > recordings.MAX_BYTES:
            results.append({"file": name, "error": "Over 100 MB"})
            continue
        t = times[i] if i < len(times) else ""
        try:
            r = recordings.save_upload({"rm_id": rm_id, "id": None}, data, name, int(t) if t.isdigit() else None,
                                       None, direction, other_number, f"manual: {actor.get('name') or ''}"[:64])
            results.append({"file": name, **r})
        except ValueError as e:
            results.append({"file": name, "error": str(e)})
    return {"results": results}


DOWNLOADS = {"transcript.txt": ("transcript_file", "txt", "text/plain; charset=utf-8"),
             "summary.md": ("summary_file", "md", "text/markdown; charset=utf-8"),
             "report.docx": ("docx_file", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
             "report.doc": ("docx_file", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}   # old links get the .docx too


@app.get("/api/recordings/{rec_id}/download/{kind}", dependencies=[Depends(allow("admin", "ops", "rm"))])
def download_recording(rec_id: int, kind: str, request: Request):
    """The transcript (.txt), the summary (.md) or both as a Word document (.docx), as a file to save."""
    if kind not in DOWNLOADS:
        raise HTTPException(404, "Choose transcript.txt, summary.md or report.docx")
    r = _own_recording(request, rec_id)
    fn, ext, media = DOWNLOADS[kind]
    name = report.filename(r, ext)
    return Response(getattr(report, fn)(r), media_type=media,
                    headers={"Content-Disposition": f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name)}"})


@app.post("/api/recordings/{rec_id}/summarize", dependencies=[Depends(allow(*STAFF))])
def summarize_recording(rec_id: int):
    """Write the full summary again (it is written by itself after each recording is processed)."""
    r = recordings.get_recording(rec_id)
    if not r:
        raise HTTPException(404, "Recording not found")
    if not r["transcript"]:
        raise HTTPException(400, "This recording has no transcript yet")
    db.update("recordings", rec_id, report_error=None, report_at=None)
    threading.Thread(target=recordings._write_summary, args=(rec_id,), daemon=True, name="summary").start()
    return {"started": True}


@app.post("/api/recordings/{rec_id}/reprocess", dependencies=[Depends(allow(*STAFF))])
def reprocess_recording(rec_id: int):
    if not recordings.reprocess(rec_id):
        raise HTTPException(404, "Recording not found")
    return {"ok": True}


@app.delete("/api/recordings/{rec_id}", dependencies=[Depends(allow("admin"))])
def delete_recording(rec_id: int):
    if not recordings.delete_recording(rec_id):
        raise HTTPException(404, "Recording not found")
    return {"ok": True}


@app.post("/api/recordings/process", dependencies=[Depends(allow(*STAFF))])
def process_recordings():
    """Process waiting recordings now (normally they are processed as they arrive)."""
    threading.Thread(target=recordings.process_pending, daemon=True, name="recordings-now").start()
    return {"started": True, "waiting": recordings.summary_counts().get("new", 0)}


# ------------------------------------------------------------------ integrations (app/integrations.py, INTEGRATION.md)
class KeyBody(BaseModel):
    name: str


class WebhookBody(BaseModel):
    url: str
    events: list[str] = ["*"]


def _who(actor: dict) -> str:
    return actor.get("name") or actor.get("email") or ""


@app.get("/api/integrations/keys", dependencies=[Depends(allow("admin"))])
def api_keys():
    return integrations.list_keys()


@app.post("/api/integrations/keys")
def new_api_key(body: KeyBody, actor=Depends(allow("admin"))):
    try:
        return integrations.create_key(body.name, _who(actor))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/integrations/keys/{key_id}", dependencies=[Depends(allow("admin"))])
def revoke_api_key(key_id: int):
    if not integrations.revoke_key(key_id):
        raise HTTPException(404, "No active key with that id")
    return {"ok": True}


@app.get("/api/integrations/webhooks", dependencies=[Depends(allow("admin"))])
def webhooks():
    return {"events": integrations.EVENTS, "webhooks": integrations.list_webhooks()}


@app.post("/api/integrations/webhooks")
def new_webhook(body: WebhookBody, actor=Depends(allow("admin"))):
    try:
        return integrations.create_webhook(body.url, body.events, _who(actor))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/integrations/webhooks/{hook_id}", dependencies=[Depends(allow("admin"))])
def remove_webhook(hook_id: int):
    if not integrations.remove_webhook(hook_id):
        raise HTTPException(404, "No active webhook with that id")
    return {"ok": True}


@app.post("/api/integrations/webhooks/{hook_id}/test", dependencies=[Depends(allow("admin"))])
def test_webhook(hook_id: int):
    if not integrations.send_test(hook_id):
        raise HTTPException(404, "No active webhook with that id")
    return {"queued": True}


@app.get("/api/integrations/deliveries", dependencies=[Depends(allow("admin"))])
def deliveries(webhook_id: int | None = None, limit: int = 100):
    return integrations.list_deliveries(webhook_id, limit)


# ------------------------------------------------------------------ simulator
class Simulate(BaseModel):
    calls: int = 10
    live: bool = True        # True: calls play out in real time so they appear as "ongoing"
    engine: str = "rules"    # rules = no LLM calls; auto = use the configured LLM if a key is present


@app.post("/api/simulate", dependencies=[Depends(allow(*STAFF))])
def simulate(body: Simulate):
    from .seed import has_real_data
    from .simulator import run_live, run_backfill
    if has_real_data():
        raise HTTPException(409, "The simulator only runs on the demo database, never next to real CSPs")
    n = max(1, min(body.calls, 50))
    if body.live:
        threading.Thread(target=run_live, args=(n, body.engine), daemon=True).start()
        return {"started": n, "mode": "live"}
    return run_backfill(n, days=0, engine=body.engine)
