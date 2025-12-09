# backend/reply_generator.py

import json
import os
from typing import Dict, Any, Optional
import textwrap

TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "templates.json",
)


def load_templates(path: str = TEMPLATES_PATH) -> Dict[str, Any]:
    """Load reply templates from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Templates file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    return templates


TEMPLATES = load_templates()


def summarize_email(text: str, max_chars: int = 200) -> str:
    """Simple heuristic summary of the email text."""
    text = " ".join(text.split())  # clean whitespace
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _fallback_generic() -> str:
    """Generic reply if templates are missing."""
    return (
        "Hi {{name}},\n\n"
        "Thanks for reaching out. We’ve received your message and will review it shortly.\n\n"
        "Best regards,\nSupport Team"
    )


def choose_template(label: str, tone: str = "formal") -> str:
    """
    Pick a template for the given label + tone.
    - Expects templates.json in the form: {label: {tone: [templates...]}}
    - Falls back gracefully if something is missing.
    """
    tone = (tone or "formal").lower()
    if tone not in {"formal", "friendly"}:
        tone = "formal"

    # 1) try requested label
    label_templates = TEMPLATES.get(label)

    # 2) fall back to "other"
    if label_templates is None:
        label_templates = TEMPLATES.get("other")

    if label_templates is None:
        return _fallback_generic()

    # If the JSON is still the old structure (list of strings), handle that
    if isinstance(label_templates, list):
        return label_templates[0] if label_templates else _fallback_generic()

    # New structure: dict of tones
    if isinstance(label_templates, dict):
        tone_templates = (
            label_templates.get(tone)
            or label_templates.get("formal")
            or next(iter(label_templates.values()), None)
        )
        if isinstance(tone_templates, list) and tone_templates:
            return tone_templates[0]

    return _fallback_generic()


def fill_template(
    template: str,
    name: str,
    email_text: str,
    label: str,
    extra_info: Optional[str] = None,
) -> str:
    """Fill placeholders in a template string."""
    summary = summarize_email(email_text)

    reply = template.replace("{{name}}", name or "Customer")
    reply = reply.replace("{{summary}}", summary)

    if extra_info is None:
        if label == "product_info":
            extra_info = (
                "Here’s a quick overview:\n"
                "- Pricing depends on usage and team size.\n"
                "- All plans include core features like dashboards, exports, and basic support.\n"
                "- Higher tiers add advanced analytics, integrations, and priority support.\n"
            )
        else:
            extra_info = ""

    reply = reply.replace("{{extra_info}}", extra_info or "")

    # 🔑 important: convert literal "\n" into real line breaks
    reply = reply.replace("\\n", "\n")

    return textwrap.dedent(reply).strip()


def generate_reply(
    email_text: str,
    label: str,
    name: Optional[str] = None,
    tone: Optional[str] = "formal",
) -> Dict[str, str]:
    """Main helper used by the FastAPI backend."""
    label = label or "other"
    tone = (tone or "formal").lower()
    if tone not in {"formal", "friendly"}:
        tone = "formal"

    template = choose_template(label, tone)

    reply_text = fill_template(
        template=template,
        name=name or "Customer",
        email_text=email_text,
        label=label,
    )

    return {
        "reply": reply_text,
        "used_label": label,
    }


if __name__ == "__main__":
    sample_email = (
        "Hi team, I'm seeing unexpected charges on my last invoice and I believe "
        "I was billed twice. Could you please review?"
    )

    out = generate_reply(sample_email, label="billing", name="Shriya", tone="friendly")
    print("\n=== Generated Reply ===\n")
    print(out["reply"])