# auto_runner.py

import os
import time
import requests

# ----------------------------------------------------
# Config
# ----------------------------------------------------
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))   # every 30 seconds
MAX_EMAILS = int(os.getenv("MAX_EMAILS", "5"))        # per poll
SEND_REPLIES = True                                   # auto-send ON


def process_once() -> int:
    """
    Call /gmail/process_unread once.
    Returns how many emails were processed.
    """
    params = {
        "max_emails": MAX_EMAILS,
        "send_replies": str(SEND_REPLIES).lower(),
    }
    url = f"{BACKEND_BASE_URL}/gmail/process_unread"
    resp = requests.post(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return int(data.get("processed_count", 0))


if __name__ == "__main__":
    print(
        f"Starting auto-runner. Poll every {POLL_SECONDS}s, "
        f"max_emails={MAX_EMAILS}, send_replies={SEND_REPLIES}"
    )
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            processed = process_once()
            print(f"[OK] processed={processed} send_replies={SEND_REPLIES}")
        except Exception as e:
            print(f"[ERROR] {e}")

        # wait before next poll
        time.sleep(POLL_SECONDS)