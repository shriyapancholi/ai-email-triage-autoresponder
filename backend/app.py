# backend/app.py

import os
import csv
from datetime import datetime
from typing import Optional, List, Dict

import re

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.classifier import EmailClassifier
from backend.reply_generator import generate_reply
from backend.email_client import fetch_unread_emails, send_email

from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------
# Paths & environment
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
LOG_PATH = os.path.join(BASE_DIR, "data", "request_log.csv")

# load environment variables from .env at project root
load_dotenv(os.path.join(BASE_DIR, ".env"))

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ----------------------------------------------------
# FastAPI setup
# ----------------------------------------------------
app = FastAPI(
    title="AI Email Triage & Autoresponder",
    description="Classifies customer emails and generates suggested replies.",
    version="1.0.0",
)

# Allow Streamlit frontend (localhost) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # later you can restrict to ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates for optional HTML page (you already have a history.html)
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ----------------------------------------------------
# Load classifier once at startup
# ----------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert_model")
MODEL_DIR = os.path.abspath(MODEL_DIR)

classifier = EmailClassifier(model_dir=MODEL_DIR)

# ----------------------------------------------------
# Request / response models
# ----------------------------------------------------
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: List[float]


class ReplyRequest(BaseModel):
    text: str
    label: str
    name: Optional[str] = None


class ReplyResponse(BaseModel):
    reply: str
    used_label: str


class ProcessEmailRequest(BaseModel):
    text: str
    name: Optional[str] = None


class ProcessEmailResponse(BaseModel):
    label: str               # predicted label
    confidence: float
    reply: str


# ----------------------------------------------------
# Logging helper
# ----------------------------------------------------
def log_request(
    *,
    name: Optional[str],
    label: str,
    confidence: float,
    email_text: str,
    reply: str,
) -> None:
    """
    Append one row to data/request_log.csv.

    We store both predicted_label and true_label so that the admin
    can later correct labels and compute accuracy.
    Initially: true_label == predicted_label.
    """
    file_exists = os.path.exists(LOG_PATH)
    fieldnames = [
        "timestamp",
        "name",
        "predicted_label",
        "true_label",
        "confidence",
        "email_text",
        "reply",
    ]

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "name": name or "",
                "predicted_label": label,
                "true_label": label,  # admin can override later
                "confidence": f"{confidence:.4f}",
                # store literal "\n" in CSV to keep rows one-line
                "email_text": email_text.replace("\n", "\\n"),
                "reply": reply.replace("\n", "\\n"),
            }
        )


# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "AI Email Triage & Autoresponder backend running"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classify an email into one of: billing, technical_issue, product_info, other.
    """
    result = classifier.predict(req.text)
    return PredictResponse(
        label=result["label"],
        confidence=float(result["confidence"]),
        probabilities=[float(p) for p in result["probabilities"]],
    )


@app.post("/generate_reply", response_model=ReplyResponse)
def generate_reply_endpoint(req: ReplyRequest):
    """
    Generate a reply given the email text + a label (e.g., from /predict).
    """
    out = generate_reply(
        email_text=req.text,
        label=req.label,
        name=req.name,
    )
    return ReplyResponse(reply=out["reply"], used_label=out["used_label"])


@app.post("/process_email", response_model=ProcessEmailResponse)
def process_email(req: ProcessEmailRequest):
    """
    Full pipeline:
    1. Classify the email.
    2. Generate a reply using that label.
    3. Log it to CSV for history dashboard.
    """
    pred = classifier.predict(req.text)
    label = pred["label"]
    confidence = float(pred["confidence"])

    reply_out = generate_reply(
        email_text=req.text,
        label=label,
        name=req.name,
    )
    reply_text = reply_out["reply"]

    # --- Log this interaction ---
    log_request(
        name=req.name,
        label=label,
        confidence=confidence,
        email_text=req.text,
        reply=reply_text,
    )

    return ProcessEmailResponse(
        label=label,
        confidence=confidence,
        reply=reply_text,
    )


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    """
    Simple HTML dashboard showing logged emails + predictions (optional).
    """
    rows: List[Dict[str, str]] = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Show latest first
    rows = rows[::-1]

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "rows": rows,
        },
    )


# ----------------------------------------------------
# Gmail integration: fetch + auto-reply to unread emails
# ----------------------------------------------------
@app.post("/gmail/process_unread")
def gmail_process_unread(
    max_emails: int = Query(5, ge=1, le=50),
    send_replies: bool = Query(True, description="If true, send replies via Gmail SMTP"),
):
    """
    1. Fetch latest unread emails from Gmail via IMAP.
    2. For each email:
       - run classification + reply generation
       - (optionally) send reply via Gmail SMTP
       - log to CSV (same log format as /process_email)
    Returns a summary of what was processed.
    """
    emails = fetch_unread_emails(max_emails=max_emails)

    results: List[Dict[str, object]] = []

    for msg in emails:
        # Use body if present, otherwise subject as text for model
        text_to_classify = (msg.get("body") or "").strip()
        if not text_to_classify:
            text_to_classify = msg.get("subject", "")

        pred = classifier.predict(text_to_classify)
        label = pred["label"]
        confidence = float(pred["confidence"])

        reply_out = generate_reply(
            email_text=text_to_classify,
            label=label,
            name=None,
        )
        reply_text = reply_out["reply"]

        # ---- log to CSV (same as manual /process_email) ----
        log_request(
            name=msg.get("from_addr"),
            label=label,
            confidence=confidence,
            email_text=text_to_classify,
            reply=reply_text,
        )

        sent_flag = False

        # ---- optionally send the email reply via SMTP ----
        if send_replies:
            raw_from = msg.get("from_addr", "")
            # Try to extract real address from: "Name <email@domain>"
            m = re.search(r"<([^>]+)>", raw_from)
            to_addr = m.group(1) if m else raw_from

            if to_addr:
                subject = msg.get("subject") or "Re: your recent message"
                subject = f"Re: {subject}"

                send_email(
                    to_addr=to_addr,
                    subject=subject,
                    body=reply_text,
                    reply_to_message_id=msg.get("message_id") or None,
                )
                sent_flag = True

        results.append(
            {
                "from": msg.get("from_addr"),
                "subject": msg.get("subject"),
                "label": label,
                "confidence": confidence,
                "sent": sent_flag,
            }
        )

    return {
        "processed_count": len(results),
        "items": results,
        "send_replies": send_replies,
    }