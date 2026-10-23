"""FastAPI app: agent API + dashboards.

Run:  uvicorn app.main:app --reload --port 8000
"""
import json
import logging
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import agent, auth, db, integrations, metrics, recordings, stt, tts  # noqa: E402
from .seed import seed  # noqa: E402
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
seed()

# ------------------------------------------------------------------ auth
PUBLIC_PATHS = {"/login", "/signup", "/api/auth/login", "/api/auth/signup", "/api/auth/rms", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)
DEVICE_PATHS = {"/api/uploader/ping", "/api/uploader/recordings"}   # the uploader app: upload token, not a login
STAFF = ("admin", "ops")                       # full access to calls, dashboards and metrics
PAGE_ROLES = {"/": ("admin", "ops", "rm"), "/call": STAFF, "/ops": STAFF, "/rm": ("admin", "ops", "rm"),
              "/users": ("admin",)}
PENDING_OK = {"/pending", "/api/auth/me", "/api/auth/logout"}   # all a not-yet-approved account can reach


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
            return JSONResponse({"detail": "Your account is waiting for admin approval"}, status_code=403)
        return RedirectResponse("pending", status_code=303)
    if role != "pending" and path == "/pending":
        return RedirectResponse("./", status_code=303)
    if path in PAGE_ROLES and role not in PAGE_ROLES[path]:
        return RedirectResponse("./?denied=1", status_code=303)
    request.state.user = user
    return await call_next(request)


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


def _signed_in(user: dict) -> JSONResponse:
    resp = JSONResponse(auth.public_user(user))
    resp.set_cookie(auth.COOKIE, auth.create_session(user["id"]), max_age=auth.SESSION_DAYS * 86400,
                    httponly=True, samesite="lax", secure=auth.COOKIE_SECURE, path="/")
    return resp


@app.post("/api/auth/signup")
def signup(body: SignupBody):
    try:
        return _signed_in(auth.signup(body.email, body.password, body.name, body.rm_id))
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e))


@app.post("/api/auth/login")
def login(body: LoginBody):
    try:
        return _signed_in(auth.login(body.email, body.password))
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e))


@app.post("/api/auth/logout")
def logout(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.get("/api/auth/me")
def me(request: Request):
    return auth.public_user(request.state.user)


@app.get("/api/auth/rms")
def signup_rms():
    """RM profiles a new user can link to (ones that don't have an account yet)."""
    return db.q("SELECT id, name, region FROM rms WHERE id NOT IN (SELECT rm_id FROM users WHERE rm_id IS NOT NULL)"
                " ORDER BY id")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(STATIC, "favicon.ico"), media_type="image/x-icon")


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


@app.get("/signup")
def signup_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


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


# ------------------------------------------------------------------ calls
class StartCall(BaseModel):
    direction: str = "inbound"
    phone: str | None = None
    purpose: str = "support"
    lang: str = "hi"   # hi | en | ask (agent asks the caller to pick)


class Turn(BaseModel):
    text: str
    stt_confidence: float | None = None


@app.post("/api/calls", dependencies=[Depends(allow(*STAFF))])
def start_call(body: StartCall):
    return agent.start_call(body.direction, body.phone, body.purpose, lang=body.lang)


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
    return {"ok": True, "rm_id": device["rm_id"], "rm_name": device["rm_name"]}


@app.post("/api/uploader/recordings")
def uploader_upload(audio: UploadFile = File(...), started_at: int | None = Form(None),
                    duration_sec: float | None = Form(None), direction: str = Form("unknown"),
                    other_number: str = Form(""), device_id: str = Form(""), device=Depends(_device)):
    data = audio.file.read(recordings.MAX_BYTES + 1)
    if len(data) > recordings.MAX_BYTES:
        raise HTTPException(413, "Recording is over 100 MB")
    try:
        return recordings.save_upload(device, data, audio.filename or "", started_at, duration_sec, direction,
                                      other_number, device_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


class TokenBody(BaseModel):
    rm_id: int
    label: str = ""


@app.get("/api/uploader/tokens", dependencies=[Depends(allow("admin"))])
def upload_tokens():
    return recordings.list_tokens()


@app.post("/api/uploader/tokens")
def new_upload_token(body: TokenBody, actor=Depends(allow("admin"))):
    try:
        return recordings.create_token(body.rm_id, body.label, actor.get("name") or actor.get("email"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/uploader/tokens/{token_id}", dependencies=[Depends(allow("admin"))])
def revoke_upload_token(token_id: int):
    if not recordings.revoke_token(token_id):
        raise HTTPException(404, "No active token with that id")
    return {"ok": True}


@app.get("/api/recordings", dependencies=[Depends(allow(*STAFF))])
def list_recordings(status: str | None = None, rm_id: int | None = None, limit: int = 200):
    return {"counts": recordings.summary_counts(), "recordings": recordings.list_recordings(status, rm_id, limit)}


@app.get("/api/recordings/{rec_id}", dependencies=[Depends(allow(*STAFF))])
def get_recording(rec_id: int):
    r = recordings.get_recording(rec_id)
    if not r:
        raise HTTPException(404, "Recording not found")
    return r


@app.post("/api/recordings/process", dependencies=[Depends(allow(*STAFF))])
def process_recordings():
    """Run the daily processing now instead of waiting for RECORDINGS_DAILY_AT."""
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
    from .simulator import run_live, run_backfill
    n = max(1, min(body.calls, 50))
    if body.live:
        threading.Thread(target=run_live, args=(n, body.engine), daemon=True).start()
        return {"started": n, "mode": "live"}
    return run_backfill(n, days=0, engine=body.engine)
