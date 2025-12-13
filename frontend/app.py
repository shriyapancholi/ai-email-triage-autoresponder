# frontend/app.py

import os
import json
from datetime import datetime, date
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------
# Paths & backend config
# ----------------------------------------------------
BACKEND_BASE_URL = "http://127.0.0.1:8000"
BACKEND_API_URL = f"{BACKEND_BASE_URL}/process_email"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")

LOG_PATH = os.path.join(DATA_DIR, "request_log.csv")
TEMPLATES_PATH = os.path.join(DATA_DIR, "templates.json")
USERS_PATH = os.path.join(DATA_DIR, "users.json")

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="AI Email Triage & Autoresponder",
    layout="wide",
)

# ----------------------------------------------------
# Auth config (demo)
# ----------------------------------------------------
ADMIN_SIGNUP_SECRET = "supersecretkey"  # <<< CHANGE THIS for your project

# Default users created only if users.json is missing
DEFAULT_USERS: Dict[str, Dict[str, str]] = {
    "admin": {"password": "admin123", "role": "admin"},
    "demo": {"password": "user123", "role": "customer"},
}


def load_users() -> Dict[str, Dict[str, str]]:
    """Load users from users.json, create defaults if missing/empty."""
    if not os.path.exists(USERS_PATH):
        save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()

    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            save_users(DEFAULT_USERS)
            return DEFAULT_USERS.copy()

        clean: Dict[str, Dict[str, str]] = {}
        for username, info in data.items():
            if not isinstance(info, dict):
                continue
            pwd = str(info.get("password", ""))
            role = str(info.get("role", "customer"))
            clean[username] = {"password": pwd, "role": role}
        if not clean:
            save_users(DEFAULT_USERS)
            return DEFAULT_USERS.copy()
        return clean
    except Exception:
        # if file is corrupted, fall back to defaults
        save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()


def save_users(users: Dict[str, Dict[str, str]]) -> None:
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------
# Helpers: logs & backend
# ----------------------------------------------------
@st.cache_data
def load_logs(path: str) -> Optional[pd.DataFrame]:
    """Read the request_log.csv created by the backend."""
    if not os.path.exists(path):
        return None

    # Read everything as string first
    df = pd.read_csv(path, dtype=str)

    # Unescape newlines
    for col in ["email_text", "reply"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.replace("\\n", "\n")

    # Confidence as numeric
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # --- NEW: normalise / create 'sent' column ---
    if "sent" not in df.columns:
        # Old logs before we added the column → treat as not sent
        df["sent"] = False
    else:
        # Convert to clean boolean
        df["sent"] = (
            df["sent"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    # Optional: normalise 'source' so filters work later
    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)

    return df


def call_backend(email_text: str, name: Optional[str]) -> dict:
    payload: Dict[str, Any] = {"text": email_text}
    if name and name.strip():
        payload["name"] = name.strip()

    resp = requests.post(BACKEND_API_URL, json=payload)
    resp.raise_for_status()
    return resp.json()


def call_gmail_process_unread(max_emails: int, send_replies: bool) -> dict:
    """
    Call backend /gmail/process_unread and return JSON.
    """
    params = {"max_emails": max_emails, "send_replies": str(send_replies).lower()}
    resp = requests.post(f"{BACKEND_BASE_URL}/gmail/process_unread", params=params)
    resp.raise_for_status()
    return resp.json()


# ----------------------------------------------------
# Helpers: templates.json
# ----------------------------------------------------
@st.cache_data
def load_templates(path: str) -> Dict[str, Any]:
    """
    Load templates as-is from JSON.

    Supports both:
    - old format: { "billing": ["template1", "template2"] }
    - new format: { "billing": { "formal": [...], "friendly": [...] }, ... }
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def save_templates(templates: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
    load_templates.clear()


# ----------------------------------------------------
# Auth views: login + signup
# ----------------------------------------------------
def show_auth():
    st.title("🔐 Login / Sign up")

    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    # ----- LOGIN -----
    with tab_login:
        st.write(
            "- Use **admin** accounts for analytics, template editing, system health.\n"
            "- Use **customer** accounts for basic triage & playground."
        )
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            users = load_users()
            user = users.get(username)
            if user and password == user["password"]:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = user["role"]
                st.success(f"Logged in as {username} ({user['role']})")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # ----- SIGN UP -----
    with tab_signup:
        st.write(
            "Create a new account. To register as **admin**, you must provide the "
            "correct admin secret key."
        )

        with st.form("signup_form"):
            new_username = st.text_input("New username")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input(
                "Confirm password", type="password"
            )
            role = st.selectbox(
                "Role",
                options=[("customer", "Customer (recommended)"),
                         ("admin", "Admin (requires secret key)")],
                format_func=lambda x: x[1],
            )[0]

            admin_key = ""
            if role == "admin":
                admin_key = st.text_input(
                    "Admin secret key (required for admin accounts)",
                    type="password",
                )

            submitted_signup = st.form_submit_button("Create account")

        if submitted_signup:
            users = load_users()

            # basic validations
            if not new_username.strip():
                st.error("Username cannot be empty.")
                return
            if " " in new_username:
                st.error("Username cannot contain spaces.")
                return
            if new_username in users:
                st.error("That username is already taken.")
                return
            if not new_password:
                st.error("Password cannot be empty.")
                return
            if new_password != confirm_password:
                st.error("Passwords do not match.")
                return

            if role == "admin":
                if admin_key != ADMIN_SIGNUP_SECRET:
                    st.error("Invalid admin secret key. Cannot create admin user.")
                    return

            # create user
            users[new_username] = {
                "password": new_password,
                "role": role,
            }
            save_users(users)

            st.success(f"Account created for `{new_username}` as `{role}` ✨")

            # auto-login after signup
            st.session_state["authenticated"] = True
            st.session_state["username"] = new_username
            st.session_state["role"] = role
            st.rerun()


# ----------------------------------------------------
# CUSTOMER DASHBOARD
# ----------------------------------------------------
def customer_dashboard():
    st.title("📧 AI Email Triage & Autoresponder")

    st.write(
        "Paste a customer email below. The system will classify it "
        "and generate a suggested reply using your fine-tuned BERT model."
    )

    with st.form("email_form"):
        name = st.text_input("Customer Name (optional)", placeholder="Shriya")

        email_text = st.text_area(
            "Customer email",
            height=220,
            placeholder="Hi team, I was billed twice on my last invoice. Please check this.",
        )

        submitted = st.form_submit_button("Analyze & Generate Reply")

    if submitted:
        if not email_text.strip():
            st.error("Please paste an email first.")
        else:
            with st.spinner("Contacting backend API..."):
                try:
                    data = call_backend(email_text=email_text, name=name)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                else:
                    st.success("Prediction received!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Detected Category")
                        st.write(f"**Label:** `{data['label']}`")
                        st.write(f"**Confidence:** {data['confidence']:.3f}")
                        if name.strip():
                            st.write(f"**Name:** {name.strip()}")

                    with col2:
                        st.subheader("Suggested Reply")
                        reply_text = data["reply"]
                        st.code(reply_text)

                        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        safe_name = (
                            name.strip().replace(" ", "_")
                            if name.strip()
                            else "customer"
                        )
                        filename = f"reply_{data['label']}_{safe_name}_{timestamp}.txt"

                        st.download_button(
                            label="📥 Download reply as .txt",
                            data=reply_text,
                            file_name=filename,
                            mime="text/plain",
                        )


# ----------------------------------------------------
# ADMIN DASHBOARD
# ----------------------------------------------------
def admin_dashboard():
    st.title("📧 AI Email Triage & Autoresponder")
    st.subheader("Admin Dashboard – Analytics & History")

    df_full = load_logs(LOG_PATH)

    if df_full is None or df_full.empty:
        st.info("No logs found yet. Ask customers to send some emails first!")
        return

    # Work on a copy
    df = df_full.copy()

    # ---------- Parse timestamp ----------
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp_dt"] = pd.NaT

    # ---------- Top metrics ----------
    total_emails = len(df)

    if "name" in df.columns:
        unique_customers = (
            df["name"].fillna("").replace("", pd.NA).nunique()
        )
    else:
        unique_customers = 0

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        avg_conf = float(df["confidence"].mean(skipna=True))
    else:
        avg_conf = None

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Emails Processed", total_emails)
    m2.metric("Unique Customers", unique_customers)
    m3.metric("Average Confidence", f"{avg_conf:.3f}" if avg_conf is not None else "N/A")

    st.markdown("---")

    # =====================================================
    # AUTO REPLIES SUMMARY  (sent emails only)
    # =====================================================
    st.subheader("Auto replies summary (sent emails only)")

    if "sent" not in df.columns:
        st.info(
            "No **sent** column found in logs – cannot distinguish auto replies.\n\n"
            "Make sure the backend logger writes a boolean column like `sent` "
            "when an email reply is actually sent."
        )
    else:
        sent_df = df[df["sent"] == True]

        if sent_df.empty:
            st.info("No auto-sent replies in the logs yet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total auto replies sent", len(sent_df))

            with c2:
                label_col = "predicted_label" if "predicted_label" in sent_df.columns else "label"
                label_series = (
                    sent_df.get(label_col, pd.Series(dtype=str))
                    .fillna("unknown")
                    .astype(str)
                    .value_counts()
                )
                st.write("By label:")
                for lbl, cnt in label_series.items():
                    st.write(f"- **{lbl}**: {int(cnt)} emails")

            st.markdown("#### Recent auto-sent replies")

            recent = sent_df.sort_values("timestamp_dt", ascending=False).head(20)

            # Show as cards / expanders instead of a PyArrow table
            for idx, row in recent.iterrows():
                ts = str(row.get("timestamp", ""))
                name = str(row.get("name", "") or row.get("from_addr", ""))
                label = str(row.get("predicted_label", row.get("label", "")))
                try:
                    conf_val = float(row.get("confidence", 0.0))
                    conf_str = f"{conf_val:.3f}"
                except Exception:
                    conf_str = "N/A"

                header_parts = []
                if ts:
                    header_parts.append(ts)
                if name:
                    header_parts.append(name)
                if label:
                    header_parts.append(f"[{label}]")
                header_parts.append(f"conf={conf_str}")
                header = " | ".join(header_parts)

                with st.expander(header):
                    st.write(f"**Timestamp:** {ts}")
                    st.write(f"**Name / From:** {name}")
                    st.write(f"**Label:** `{label}`")
                    st.write(f"**Confidence:** {conf_str}")
                    if "source" in row:
                        st.write(f"**Source:** {row['source']}")

                    st.markdown("---")
                    if "email_text" in row:
                        st.markdown("**Email text:**")
                        st.write(str(row["email_text"]).replace("\\n", "\n"))

                    if "reply" in row:
                        st.markdown("**Reply sent:**")
                        st.write(str(row["reply"]).replace("\\n", "\n"))

    st.markdown("---")

    # =====================================================
    # FILTERS (all processed emails)
    # =====================================================
    st.subheader("Filters (all processed emails)")

    df_filtered = df.copy()

    # Label filter
    label_col = "predicted_label" if "predicted_label" in df_filtered.columns else "label"
    if label_col in df_filtered.columns:
        label_options = (
            sorted(df_filtered[label_col].fillna("unknown").astype(str).unique().tolist())
        )
    else:
        label_options = []

    selected_labels = st.multiselect(
        "Filter by label",
        options=label_options,
        default=label_options,
    )

    if selected_labels and label_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[label_col].isin(selected_labels)]

    # Name filter
    name_filter = st.text_input(
        "Filter by customer name (contains)",
        placeholder="e.g., akshat",
    ).strip()

    if name_filter and "name" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["name"].fillna("").str.contains(name_filter, case=False)
        ]

    # Confidence slider
    if "confidence" in df_filtered.columns:
        min_conf_present = df_filtered["confidence"].min(skipna=True)
        min_conf = float(min_conf_present if pd.notna(min_conf_present) else 0.0)
    else:
        min_conf = 0.0

    min_conf_slider = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=min_conf,
        step=0.01,
    )

    if "confidence" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["confidence"].fillna(0) >= min_conf_slider
        ]

    # Date range filter
    if df_filtered["timestamp_dt"].notna().any():
        min_date = df_filtered["timestamp_dt"].min().date()
        max_date = df_filtered["timestamp_dt"].max().date()
        start_date, end_date = st.date_input(
            "Date range (UTC)",
            value=(min_date, max_date),
        )
        if isinstance(start_date, date) and isinstance(end_date, date):
            mask = (df_filtered["timestamp_dt"].dt.date >= start_date) & (
                df_filtered["timestamp_dt"].dt.date <= end_date
            )
            df_filtered = df_filtered[mask]

    st.write(f"Showing {len(df_filtered)} of {len(df)} emails after filters.")

    csv_bytes = df_filtered.drop(
        columns=["timestamp_dt"], errors="ignore"
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        "📄 Download filtered logs as CSV",
        data=csv_bytes,
        file_name="email_logs_filtered.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # =====================================================
    # RECENT EMAILS DETAIL VIEW
    # =====================================================
    st.subheader("Recent Emails")

    df_display = df_filtered.copy()

    for col in ["timestamp", "name", label_col, "email_text", "reply"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].fillna("").astype(str)

    if "timestamp_dt" in df_display.columns:
        df_display = df_display.sort_values("timestamp_dt", ascending=False)

    max_rows = 50
    for idx, row in df_display.head(max_rows).iterrows():
        title_parts = []
        if row.get("timestamp"):
            title_parts.append(row["timestamp"])
        if row.get("name"):
            title_parts.append(row["name"])
        if row.get(label_col):
            title_parts.append(f"[{row[label_col]}]")
        if pd.notna(row.get("confidence")):
            try:
                conf_val = float(row["confidence"])
                title_parts.append(f"conf={conf_val:.3f}")
            except Exception:
                pass

        header = " | ".join(title_parts) if title_parts else f"Email #{idx}"

        with st.expander(header):
            st.markdown("**Email text:**")
            st.write(row.get("email_text", "").replace("\\n", "\n"))

            st.markdown("---")
            st.markdown("**Reply:**")
            st.write(row.get("reply", "").replace("\\n", "\n"))

# ----------------------------------------------------
# SIMPLE PLAYGROUND (single email, no Gmail)
# ----------------------------------------------------
def playground_view():
    st.title("🧪 Playground")

    st.write(
        "Try out different customer emails and see how the system responds. "
        "This is a sandbox and does **not** affect the stored logs."
    )

    with st.form("playground_form"):
        email_text = st.text_area(
            "Customer email",
            height=220,
            placeholder="Hi team, My payment failed yesterday but the amount was deducted from my bank. Can you resolve this?",
        )
        submitted = st.form_submit_button("Run playground")

    if submitted:
        if not email_text.strip():
            st.error("Please paste an email first.")
        else:
            with st.spinner("Calling backend /process_email..."):
                try:
                    data = call_backend(email_text=email_text, name=None)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                else:
                    st.success("Response received from backend!")

                    st.subheader("Model classification")
                    st.write(f"- **Label:** `{data['label']}`")
                    st.write(f"- **Confidence:** {data['confidence']:.3f}")

                    st.subheader("Suggested reply")
                    st.code(data["reply"])


# ----------------------------------------------------
# TEMPLATE EDITOR (admin only)
# ----------------------------------------------------
def template_editor_view():
    st.title("📝 Template Editor")

    st.write(
        "Edit the reply templates stored in `data/templates.json`.\n\n"
        "Placeholders:\n"
        "- `{{name}}`\n"
        "- `{{summary}}`\n"
        "- `{{extra_info}}`\n\n"
        "**Template indexes are 0-based.**"
    )

    # Load templates from JSON
    templates = load_templates(TEMPLATES_PATH)
    if not templates:
        st.warning("No templates found. Creating an empty structure.")
        templates = {}

    # -----------------------------
    # Choose label
    # -----------------------------
    labels = sorted(templates.keys())
    if not labels:
        st.info("No labels found. Add a new label below.")
        new_label = st.text_input("New label name (e.g., billing)")
        if st.button("Create label"):
            if new_label.strip():
                templates[new_label.strip()] = [""]
                save_templates(templates, TEMPLATES_PATH)
                st.success(f"Label '{new_label.strip()}' created.")
                st.rerun()
        return

    label = st.selectbox("Choose label", options=labels)

    label_data = templates.get(label, [])

    # Detect whether this label uses multi-tone format or simple list
    if isinstance(label_data, dict):
        is_multi_tone = True
        tones = sorted(label_data.keys())
        tone = st.selectbox("Tone", options=tones)
        current_list = label_data.get(tone, [])
    else:
        is_multi_tone = False
        tone = None
        current_list = label_data if isinstance(label_data, list) else []

    # -----------------------------
    # Show all existing templates
    # -----------------------------
    st.markdown("### Existing templates")

    if is_multi_tone:
        for t_name, t_list in label_data.items():
            st.markdown(f"**Tone: `{t_name}`**")
            if t_list:
                for idx, tpl in enumerate(t_list):
                    st.code(f"[{idx}] {tpl}", language="markdown")
            else:
                st.info(f"No templates under tone `{t_name}`.")
    else:
        if current_list:
            for idx, tpl in enumerate(current_list):
                st.code(f"[{idx}] {tpl}", language="markdown")
        else:
            st.info("No templates for this label yet.")

    # -----------------------------
    # Edit a specific template
    # -----------------------------
    st.caption(
        f"This label currently has {len(current_list)} template(s). "
        "Indexes are 0..(count - 1)."
    )

    max_index = max(len(current_list) - 1, 0)
    tmpl_idx = st.number_input(
        "Template index (0-based)",
        min_value=0,
        max_value=max_index,
        step=1,
        value=0,
    )

    current_text = ""
    if 0 <= int(tmpl_idx) < len(current_list):
        current_text = current_list[int(tmpl_idx)]

    tmpl_text = st.text_area(
        "Template text",
        value=current_text,
        height=220,
    )

    col_save, col_add, col_delete = st.columns(3)

    # -----------------------------
    # Save / Add / Delete actions
    # -----------------------------
    with col_save:
        if st.button("💾 Save changes to this template"):
            current_list[int(tmpl_idx)] = tmpl_text

            if is_multi_tone:
                label_data[tone] = current_list
                templates[label] = label_data
            else:
                templates[label] = current_list

            save_templates(templates, TEMPLATES_PATH)
            st.success("Template updated.")
            st.rerun()

    with col_add:
        if st.button("➕ Add as new template (append)"):
            current_list.append(tmpl_text)

            if is_multi_tone:
                label_data[tone] = current_list
                templates[label] = label_data
            else:
                templates[label] = current_list

            save_templates(templates, TEMPLATES_PATH)
            st.success("New template added.")
            st.rerun()

    with col_delete:
        if st.button("🗑️ Delete this template"):
            if len(current_list) > 1:
                del current_list[int(tmpl_idx)]
                if is_multi_tone:
                    label_data[tone] = current_list
                    templates[label] = label_data
                else:
                    templates[label] = current_list
            else:
                # If it was the last template, remove tone/label
                if is_multi_tone:
                    del label_data[tone]
                    if not label_data:  # no tones left
                        del templates[label]
                    else:
                        templates[label] = label_data
                else:
                    del templates[label]

            save_templates(templates, TEMPLATES_PATH)
            st.success("Template (or label) deleted.")
            st.rerun()


# ----------------------------------------------------
# SYSTEM HEALTH (admin only)
# ----------------------------------------------------
def system_health_view():
    st.title("🩺 System Health")

    st.subheader("Backend connectivity")

    cols = st.columns(2)
    with cols[0]:
        st.write("**Backend base URL:**", BACKEND_BASE_URL)
        with st.spinner("Pinging backend `/` ..."):
            try:
                resp = requests.get(BACKEND_BASE_URL + "/", timeout=5)
                st.success(f"OK · {resp.status_code}")
                try:
                    st.json(resp.json())
                except Exception:
                    st.write(resp.text[:300])
            except Exception as e:
                st.error(f"Backend not reachable: {e}")

    with cols[1]:
        with st.spinner("Testing `/process_email` with a dummy email..."):
            try:
                test_payload = {
                    "text": "Hi team, this is a test email to check system health."
                }
                resp = requests.post(
                    BACKEND_API_URL, json=test_payload, timeout=8
                )
                resp.raise_for_status()
                st.success("OK · /process_email responded")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Health check failed: {e}")

    st.markdown("---")

    st.subheader("Data & templates")

    logs_exist = os.path.exists(LOG_PATH)
    st.write(
        f"**Logs file:** `{LOG_PATH}` – {'✅ found' if logs_exist else '❌ not found'}"
    )
    if logs_exist:
        df = load_logs(LOG_PATH)
        if df is not None:
            st.write(f"- Rows in log: **{len(df)}**")

    templates_exist = os.path.exists(TEMPLATES_PATH)
    st.write(
        f"**Templates file:** `{TEMPLATES_PATH}` – "
        f"{'✅ found' if templates_exist else '❌ not found'}"
    )
    if templates_exist:
        tmpl = load_templates(TEMPLATES_PATH)
        st.write(f"- Labels: {', '.join(sorted(tmpl.keys())) or 'none'}")


# ----------------------------------------------------
# MAIN APP ROUTER (after login)
# ----------------------------------------------------
def main_app():
    role = st.session_state.get("role", "customer")
    username = st.session_state.get("username", "unknown")

    st.sidebar.title("Select Dashboard")
    st.sidebar.markdown(
        f"**User:** `{username}`  \n"
        f"**Role:** `{role}`"
    )

    # Logout
    if st.sidebar.button("🚪 Logout"):
        for key in ("authenticated", "username", "role"):
            st.session_state.pop(key, None)
        st.rerun()

    # Role-based options
    if role == "admin":
        options = [
            "Customer Dashboard",
            "Admin Dashboard",
            "Playground",
            "Template Editor",
            "System Health",
        ]
    else:
        options = [
            "Customer Dashboard",
            "Playground",
        ]

    view = st.sidebar.radio("Choose a view", options)

    # ---------------------------
    # CUSTOMER DASHBOARD
    # ---------------------------
    if view == "Customer Dashboard":
        customer_dashboard()

    # ---------------------------
    # ADMIN DASHBOARD
    # ---------------------------
    elif view == "Admin Dashboard":
        if role != "admin":
            st.error("You do not have permission to view the admin dashboard.")
        else:
            admin_dashboard()

    # ---------------------------
    # PLAYGROUND (Gmail)
    # ---------------------------
    elif view == "Playground":
        st.title("🧪 Gmail Playground")

        st.write(
            "Use this page to pull **unread Gmail messages**, classify them with your BERT model, "
            "generate replies, and optionally **auto-send** those replies via Gmail."
        )

        st.markdown("### Settings")
        col_a, col_b = st.columns(2)

        with col_a:
            max_emails = st.number_input(
                "Max unread emails to process",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
            )

        with col_b:
            send_mode = st.radio(
                "Mode",
                ["Preview only (no send)", "Process & send replies"],
                index=0,
            )

        send_replies = send_mode == "Process & send replies"

        if st.button("Run Gmail processing"):
            with st.spinner("Processing unread emails…"):
                try:
                    result = call_gmail_process_unread(
                        max_emails=int(max_emails),
                        send_replies=send_replies,
                    )

                    st.success(
                        f"Processed {result.get('processed_count', 0)} email(s). "
                        + (
                            "Replies were sent ✅"
                            if send_replies
                            else "No replies sent (preview mode) 🧐"
                        )
                    )

                    items = result.get("items", [])
                    if not items:
                        st.info("No unread emails were found.")
                    else:
                        st.markdown("### Results")

                        # ---- Build rows for display ----
                        summary_rows = []
                        for item in items:
                            summary_rows.append(
                                {
                                    "From": str(item.get("from", "")),
                                    "Subject": str(item.get("subject", "")),
                                    "Label": str(item.get("label", "")),
                                    "Confidence": float(item.get("confidence", 0.0)),
                                    "Sent": "Yes" if item.get("sent") else "No",
                                }
                            )

                        # ---- Render as Markdown table (no Arrow, no errors) ----
                        header = "| From | Subject | Label | Confidence | Sent |\n"
                        header += "| --- | --- | --- | --- | --- |\n"

                        rows_md = []
                        for r in summary_rows:
                            rows_md.append(
                                f"| {r['From']} | {r['Subject']} | {r['Label']} "
                                f"| {r['Confidence']:.3f} | {r['Sent']} |"
                            )

                        table_md = header + "\n".join(rows_md)
                        st.markdown(table_md)

                        st.markdown("**Raw JSON response from backend:**")
                        st.json(result)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Set options above and click **Run Gmail processing** to start.")

    # ---------------------------
    # TEMPLATE EDITOR
    # ---------------------------
    elif view == "Template Editor":
        if role != "admin":
            st.error("You do not have permission to edit templates.")
        else:
            template_editor_view()

    # ---------------------------
    # SYSTEM HEALTH
    # ---------------------------
    elif view == "System Health":
        if role != "admin":
            st.error("You do not have permission to view system health.")
        else:
            system_health_view()

# ----------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------
if __name__ == "__main__":
    if not st.session_state.get("authenticated"):
        show_auth()
    else:
        main_app()