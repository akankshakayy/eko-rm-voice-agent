"""Sends account emails (password reset, email verification) over SMTP.

Off until SMTP_HOST, SMTP_FROM and APP_BASE_URL are set. Links in emails are always built from APP_BASE_URL,
never from the request's Host header, so a forged Host can't redirect a reset link to another site.
Emails carry only a link and the account's own name; never CSP data.
"""
import logging
import os
import smtplib
import ssl
import threading
from email.message import EmailMessage

log = logging.getLogger("eko.mail")


def base_url() -> str:
    return os.environ.get("APP_BASE_URL", "").strip().rstrip("/")


def configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM") and base_url())


def link(page: str, token: str) -> str:
    return f"{base_url()}/{page}?token={token}"


def _deliver(to: str, subject: str, text: str):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = os.environ["SMTP_FROM"], to, subject
    msg.set_content(text)
    host, port = os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587"))
    security = os.environ.get("SMTP_SECURITY", "starttls").lower()   # starttls | ssl | none
    ctx = ssl.create_default_context()
    smtp = smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) if security == "ssl" else \
        smtplib.SMTP(host, port, timeout=20)
    with smtp:
        if security == "starttls":
            smtp.starttls(context=ctx)
        if os.environ.get("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        smtp.send_message(msg)


def send(to: str, subject: str, text: str) -> bool:
    """Sends in the background, so a reply never takes longer for an address that has an account."""
    if not configured():
        return False

    def run():
        try:
            _deliver(to, subject, text)
        except Exception as e:   # the address and link stay out of the log
            log.error("Account email could not be sent: %s", type(e).__name__)
    _spawn(run)
    return True


def _spawn(fn):
    threading.Thread(target=fn, daemon=True, name="mail").start()
