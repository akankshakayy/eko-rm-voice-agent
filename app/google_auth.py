"""Sign in with Google, for company accounts only.

Google proves the person owns the email address, so an account made this way can be used straight away, with no
admin approval and no confirmation email. Only verified addresses in ALLOWED_EMAIL_DOMAINS are accepted; anything
else is refused, whatever Google says.

The flow is OAuth 2.0 authorization code with PKCE. /api/auth/google/start sends the browser to Google, with a
random `state` and a PKCE code verifier kept in a short-lived HttpOnly cookie. Google sends it back to
/api/auth/google/callback, which checks that `state` matches, exchanges the code server-to-server for an access
token, and reads the user's verified email from Google's userinfo endpoint over HTTPS.

Set up once in Google Cloud Console (APIs & Services > Credentials > OAuth client ID, type "Web application"):
    authorised redirect URI: <APP_BASE_URL>/api/auth/google/callback
        e.g. https://indev.eko.in/rm-voice-agent/api/auth/google/callback, or http://localhost:8030/api/auth/google/callback
    GOOGLE_CLIENT_ID=...   GOOGLE_CLIENT_SECRET=...   APP_BASE_URL=...   (in .env)
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from urllib.parse import urlencode

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
COOKIE = "eko_google"
FLOW_SECONDS = 600


class GoogleError(Exception):
    pass


def _secret() -> bytes:
    """A per-install secret for signing the flow cookie (other apps on a shared host could set cookies too).
    SESSION_SECRET if set, else a random one kept in data/ next to the database."""
    if os.environ.get("SESSION_SECRET"):
        return os.environ["SESSION_SECRET"].encode()
    from . import db
    path = os.path.join(os.path.dirname(os.path.abspath(db.DB_PATH)), ".session_secret")
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip().encode()


def _sign(data: str) -> str:
    return data + "." + hmac.new(_secret(), data.encode(), hashlib.sha256).hexdigest()


def _unsign(value: str) -> str | None:
    data, _, mac = (value or "").rpartition(".")
    good = hmac.new(_secret(), data.encode(), hashlib.sha256).hexdigest()
    return data if data and hmac.compare_digest(mac, good) else None


def configured() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET")
                and os.environ.get("APP_BASE_URL"))


def redirect_uri() -> str:
    return os.environ["APP_BASE_URL"].rstrip("/") + "/api/auth/google/callback"


def start(next_page: str, domains: list[str]) -> tuple[str, str]:
    """-> (Google's sign-in address, the value for the flow cookie)."""
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    params = {"client_id": os.environ["GOOGLE_CLIENT_ID"], "redirect_uri": redirect_uri(), "response_type": "code",
              "scope": "openid email profile", "state": state, "code_challenge": challenge,
              "code_challenge_method": "S256", "prompt": "select_account"}
    if len(domains) == 1:
        params["hd"] = domains[0]       # Google shows only that organisation's accounts (checked again on return)
    flow = base64.urlsafe_b64encode(json.dumps({"s": state, "v": verifier, "n": next_page,
                                                "t": time.time()}).encode()).decode()
    return f"{AUTH_URL}?{urlencode(params)}", _sign(flow)


def finish(code: str, state: str, flow_cookie: str | None) -> dict:
    """Check the flow, then ask Google who signed in. -> {"email", "name", "sub", "next"}; GoogleError if anything is off."""
    try:
        flow = json.loads(base64.urlsafe_b64decode((_unsign(flow_cookie) or "").encode()))
    except (ValueError, TypeError):
        raise GoogleError("The sign-in took too long or was started in another browser. Try again.")
    if not state or not secrets.compare_digest(str(flow.get("s", "")), state) or time.time() - flow.get("t", 0) > FLOW_SECONDS:
        raise GoogleError("The sign-in took too long or was started in another browser. Try again.")
    import httpx
    tok = httpx.post(TOKEN_URL, timeout=20, data={
        "code": code, "client_id": os.environ["GOOGLE_CLIENT_ID"], "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "redirect_uri": redirect_uri(), "grant_type": "authorization_code", "code_verifier": flow["v"]})
    if tok.status_code != 200 or "access_token" not in tok.json():
        raise GoogleError("Google didn't confirm the sign-in. Try again.")
    info = httpx.get(USERINFO_URL, timeout=20, headers={"Authorization": f"Bearer {tok.json()['access_token']}"})
    if info.status_code != 200:
        raise GoogleError("Couldn't read your Google account. Try again.")
    u = info.json()
    if not u.get("email") or u.get("email_verified") is not True or not u.get("sub"):
        raise GoogleError("Your Google account's email isn't verified.")
    return {"email": u["email"].lower(), "name": u.get("name") or u["email"].split("@")[0], "sub": str(u["sub"]),
            "next": flow.get("n") or "./"}
