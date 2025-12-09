# backend/email_client.py

import os
import imaplib
import smtplib
import ssl
from typing import List, Dict, Any, Optional

from email.message import EmailMessage
from email.header import decode_header, make_header
from email import message_from_bytes

from dotenv import load_dotenv

# ----------------------------------------------------
# Load .env from the project root
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../ai-email-triage-autoresponder
ENV_PATH = os.path.join(BASE_DIR, ".env")

# This reads GMAIL_USER, GMAIL_APP_PASSWORD, etc. from .env
load_dotenv(dotenv_path=ENV_PATH)

# Credentials
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# Gmail servers (can override via .env if you ever need)
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))  # 587 = STARTTLS


def _ensure_config() -> None:
    """Make sure required Gmail config is present, otherwise raise a clear error."""
    missing = []
    if not GMAIL_USER:
        missing.append("GMAIL_USER")
    if not GMAIL_APP_PASSWORD:
        missing.append("GMAIL_APP_PASSWORD")

    if missing:
        raise RuntimeError(
            f"Missing Gmail config env vars: {', '.join(missing)}. "
            f"Tried to load from: {ENV_PATH}"
        )


def _decode_header(value: str) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def fetch_unread_emails(max_emails: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch latest UNSEEN emails from Gmail via IMAP.
    Returns list of dicts with: from_addr, subject, body, msg_id, message_id.
    """
    _ensure_config()

    # Connect to Gmail IMAP
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
    mail.select("INBOX")

    status, data = mail.search(None, "UNSEEN")
    if status != "OK":
        mail.logout()
        return []

    ids = data[0].split()
    ids = ids[-max_emails:]  # take latest N

    messages: List[Dict[str, Any]] = []

    for msg_id in ids:
        status, msg_data = mail.fetch(msg_id, "(RFC822)")
        if status != "OK":
            continue

        raw_email = msg_data[0][1]
        msg = message_from_bytes(raw_email)

        from_addr = _decode_header(msg.get("From", ""))
        subject = _decode_header(msg.get("Subject", ""))
        message_id = msg.get("Message-ID", "").strip()

        # Extract plain-text body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition") or "").lower()
                if ctype == "text/plain" and "attachment" not in disp:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body = part.get_payload(decode=True).decode(
                            charset, errors="ignore"
                        )
                    except Exception:
                        body = part.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )
                    break
        else:
            charset = msg.get_content_charset() or "utf-8"
            body = msg.get_payload(decode=True).decode(charset, errors="ignore")

        messages.append(
            {
                "imap_id": msg_id,  # raw IMAP id (bytes)
                "from_addr": from_addr,
                "subject": subject,
                "body": body,
                "message_id": message_id,
            }
        )

    mail.logout()
    return messages


def send_email(
    to_addr: str,
    subject: str,
    body: str,
    reply_to_message_id: Optional[str] = None,
) -> None:
    """
    Send an email via Gmail SMTP (STARTTLS).
    """
    _ensure_config()

    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_addr
    msg["Subject"] = subject

    if reply_to_message_id:
        msg["In-Reply-To"] = reply_to_message_id
        msg["References"] = reply_to_message_id

    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)