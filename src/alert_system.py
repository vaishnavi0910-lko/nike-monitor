"""
Brand Monitor — Alert System (Multi-User)
==========================================
Sends Email (Gmail) + SMS (Twilio) alerts to ALL registered users
based on their individual subscription preferences stored in the DB.

Each user controls:
  - alert_email        → receive email alerts (0/1)
  - alert_sms          → receive SMS alerts (0/1)
  - phone              → their phone number for SMS
  - threshold          → their personal confidence threshold (0.5–1.0)
  - bulk_count         → how many fakes trigger a bulk alert for them
  - notify_fake        → alert on high-confidence single fakes (0/1)
  - notify_bulk        → alert on bulk surge (0/1)
  - notify_suspicious  → alert on uncertain/borderline posts (0/1)
  - keywords           → extra keywords they care about (comma-separated)

Gmail App Password:
    myaccount.google.com → Security → 2-Step Verification → App Passwords

Twilio (free trial):
    twilio.com → Console → Account SID + Auth Token + Trial Number
"""

import os
import smtplib
import sqlite3
import logging
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime             import datetime, timezone, timedelta
from typing               import List, Dict, Optional, Tuple
from dotenv               import load_dotenv

load_dotenv()

logger = logging.getLogger("alerts")


# ══════════════════════════════════════════════════════════════════
# CONFIG — loaded from .env
# ══════════════════════════════════════════════════════════════════

GMAIL_SENDER       = os.getenv("EMAIL_USER", "")
GMAIL_PASSWORD     = os.getenv("EMAIL_PASS", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_PHONE", "")
BRAND_NAME         = os.getenv("BRAND_NAME", "Brand Monitor")

DEFAULT_THRESHOLD    = 0.60
DEFAULT_BULK_COUNT   = 10
BULK_WINDOW_MINUTES  = 10
COOLDOWN_MINUTES     = 15


# ══════════════════════════════════════════════════════════════════
# DATABASE — fetch subscribed users
# ══════════════════════════════════════════════════════════════════

def _get_db_path() -> str:
    from pathlib import Path
    return str(Path("data/brand_monitor.db"))

def get_users_for_alert_type(alert_type: str) -> List[Dict]:
    """
    Returns all users who have a specific alert type enabled
    AND at least one delivery channel (email or SMS) active.
    alert_type: 'fake' | 'bulk' | 'suspicious'
    """
    col_map = {
        "fake":       "notify_fake",
        "bulk":       "notify_bulk",
        "suspicious": "notify_suspicious",
    }
    col = col_map.get(alert_type, "notify_fake")
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            SELECT
                u.id, u.name, u.email, u.brand_name,
                s.alert_email, s.alert_sms, s.phone,
                s.threshold, s.bulk_count,
                s.notify_fake, s.notify_bulk, s.notify_suspicious,
                s.keywords
            FROM users u
            JOIN subscriptions s ON s.user_id = u.id
            WHERE s.{col} = 1
              AND (s.alert_email = 1 OR s.alert_sms = 1)
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"DB fetch failed for '{alert_type}': {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# ALERT STATE — per-user cooldown tracking
# ══════════════════════════════════════════════════════════════════

class AlertState:
    def __init__(self):
        self.last_high_conf: Dict[int, datetime] = {}   # user_id → datetime
        self.last_bulk:      Dict[int, datetime] = {}   # user_id → datetime
        self.recent_fakes:   List[dict]          = []
        self.alert_log:      List[dict]          = []

state = AlertState()

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _cooled_down(last: Optional[datetime]) -> bool:
    if last is None:
        return True
    t = last if last.tzinfo else last.replace(tzinfo=timezone.utc)
    return _now_utc() - t > timedelta(minutes=COOLDOWN_MINUTES)

def _parse_utc(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return _now_utc()


# ══════════════════════════════════════════════════════════════════
# SENDERS
# ══════════════════════════════════════════════════════════════════

def send_email(subject: str, html_body: str, recipients: List[str]) -> bool:
    """Send HTML email to a list of addresses via Gmail SMTP."""
    if not recipients or not GMAIL_SENDER or not GMAIL_PASSWORD:
        logger.warning("Email skipped — missing config or recipients")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{BRAND_NAME} <{GMAIL_SENDER}>"
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, recipients, msg.as_string())
        logger.info(f"Email sent: '{subject}' → {recipients}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False

def send_sms(body: str, recipients: List[str]) -> bool:
    """Send SMS to a list of phone numbers via Twilio."""
    if not recipients or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("SMS skipped — missing config or recipients")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        for number in recipients:
            if number and number.strip():
                client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=number.strip())
        logger.info(f"SMS sent → {recipients}")
        return True
    except ImportError:
        logger.error("Twilio not installed: pip install twilio")
        return False
    except Exception as e:
        logger.error(f"SMS failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# EMAIL TEMPLATES
# ══════════════════════════════════════════════════════════════════

def build_high_conf_email(post: dict, user_name: str = "") -> str:
    score_pct       = round(post.get("final_score", 0) * 100, 1)
    caption         = post.get("caption", "")
    caption_preview = (caption[:400] + "...") if len(caption) > 400 else caption
    now_str         = datetime.now().strftime("%d %b %Y, %I:%M %p")
    greeting        = f"Hi {user_name}," if user_name else "Alert,"

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;
                background:#0e0e12;color:#e8e8f0;border-radius:12px;overflow:hidden">
      <div style="background:#ff3b5c;padding:20px 24px">
        <h2 style="margin:0;color:#fff;font-size:18px">🚨 High-Confidence Counterfeit Detected</h2>
        <p style="margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:13px">
          {BRAND_NAME} Alert — {now_str}
        </p>
      </div>
      <div style="padding:20px 24px;border-bottom:1px solid #2a2a38">
        <p style="margin:0;font-size:13px;color:#c8c8d8">{greeting}</p>
        <p style="margin:8px 0 0;font-size:13px;color:#c8c8d8">
          A post with a <strong style="color:#ff3b5c">{score_pct}% counterfeit confidence
          score</strong> has been detected on your monitored feed.
        </p>
      </div>
      <div style="padding:24px;border-bottom:1px solid #2a2a38">
        <div style="display:inline-block;background:rgba(255,59,92,0.15);
                    border:1px solid #ff3b5c;border-radius:8px;padding:12px 20px">
          <div style="font-size:36px;font-weight:700;color:#ff3b5c;
                      font-family:'Courier New',monospace">{score_pct}%</div>
          <div style="font-size:11px;color:#6b6b80;margin-top:2px">COUNTERFEIT CONFIDENCE</div>
        </div>
      </div>
      <div style="padding:24px;border-bottom:1px solid #2a2a38">
        <h3 style="margin:0 0 14px;font-size:13px;color:#6b6b80;
                   text-transform:uppercase;letter-spacing:1px">Post Details</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr>
            <td style="padding:8px 0;color:#6b6b80;width:130px">Account</td>
            <td style="padding:8px 0;color:#4d8eff;font-weight:600">@{post.get("username","unknown")}</td>
          </tr>
          <tr style="border-top:1px solid #1e1e28">
            <td style="padding:8px 0;color:#6b6b80">Fake Score</td>
            <td style="padding:8px 0;color:#ff3b5c;font-weight:600">{score_pct}%</td>
          </tr>
          <tr style="border-top:1px solid #1e1e28">
            <td style="padding:8px 0;color:#6b6b80">Label</td>
            <td style="padding:8px 0">{post.get("label","fake").upper()}</td>
          </tr>
          <tr style="border-top:1px solid #1e1e28">
            <td style="padding:8px 0;color:#6b6b80">Sentiment</td>
            <td style="padding:8px 0">{post.get("sentiment","—").capitalize()}</td>
          </tr>
          <tr style="border-top:1px solid #1e1e28">
            <td style="padding:8px 0;color:#6b6b80">Likes</td>
            <td style="padding:8px 0">{post.get("likes",0):,}</td>
          </tr>
          <tr style="border-top:1px solid #1e1e28">
            <td style="padding:8px 0;color:#6b6b80">Detected At</td>
            <td style="padding:8px 0">{now_str}</td>
          </tr>
        </table>
      </div>
      <div style="padding:24px;border-bottom:1px solid #2a2a38">
        <h3 style="margin:0 0 10px;font-size:13px;color:#6b6b80;
                   text-transform:uppercase;letter-spacing:1px">Caption</h3>
        <div style="background:#1e1e28;border-left:3px solid #ff3b5c;
                    border-radius:4px;padding:14px;font-size:13px;
                    line-height:1.6;color:#c8c8d8">{caption_preview}</div>
      </div>
      <div style="padding:16px 24px;text-align:center">
        <p style="font-size:11px;color:#6b6b80;margin:0">
          {BRAND_NAME} · You received this because you enabled fake alerts in your account.
        </p>
      </div>
    </div>"""


def build_bulk_alert_email(fake_posts: List[dict], window_mins: int, user_name: str = "") -> str:
    count     = len(fake_posts)
    avg_score = sum(p.get("final_score", 0) for p in fake_posts) / max(count, 1)
    now_str   = datetime.now().strftime("%d %b %Y, %I:%M %p")
    greeting  = f"Hi {user_name}," if user_name else "Alert,"

    rows = ""
    for p in fake_posts[:8]:
        score_pct = round(p.get("final_score", 0) * 100, 1)
        caption   = str(p.get("caption", ""))[:80]
        rows += f"""
        <tr style="border-top:1px solid #1e1e28">
          <td style="padding:8px;color:#4d8eff">@{p.get("username","?")}</td>
          <td style="padding:8px;color:#ff3b5c">{score_pct}%</td>
          <td style="padding:8px;color:#c8c8d8">{caption}...</td>
        </tr>"""

    overflow = (
        f'<p style="font-size:11px;color:#6b6b80;margin-top:8px">...and {count-8} more</p>'
        if count > 8 else ""
    )

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:620px;margin:0 auto;
                background:#0e0e12;color:#e8e8f0;border-radius:12px;overflow:hidden">
      <div style="background:#ff9500;padding:20px 24px">
        <h2 style="margin:0;color:#fff;font-size:18px">⚠️ Bulk Counterfeit Surge Detected</h2>
        <p style="margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:13px">
          {BRAND_NAME} Alert — {now_str}
        </p>
      </div>
      <div style="padding:20px 24px;border-bottom:1px solid #2a2a38">
        <p style="margin:0;font-size:13px;color:#c8c8d8">{greeting}</p>
        <p style="margin:8px 0 0;font-size:13px;color:#c8c8d8">
          <strong style="color:#ff9500">{count} counterfeit posts</strong> were detected
          in the last <strong>{window_mins} minutes</strong>.
        </p>
      </div>
      <div style="padding:24px;border-bottom:1px solid #2a2a38;display:flex;gap:32px">
        <div>
          <div style="font-size:36px;font-weight:700;color:#ff9500">{count}</div>
          <div style="font-size:11px;color:#6b6b80">FAKE POSTS</div>
        </div>
        <div>
          <div style="font-size:36px;font-weight:700;color:#ff9500">{window_mins}m</div>
          <div style="font-size:11px;color:#6b6b80">TIME WINDOW</div>
        </div>
        <div>
          <div style="font-size:36px;font-weight:700;color:#ff9500">{round(avg_score*100,1)}%</div>
          <div style="font-size:11px;color:#6b6b80">AVG CONFIDENCE</div>
        </div>
      </div>
      <div style="padding:24px">
        <h3 style="margin:0 0 12px;font-size:13px;color:#6b6b80;
                   text-transform:uppercase;letter-spacing:1px">Top Flagged Posts</h3>
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <tr>
            <th style="padding:8px;text-align:left;color:#6b6b80">Account</th>
            <th style="padding:8px;text-align:left;color:#6b6b80">Score</th>
            <th style="padding:8px;text-align:left;color:#6b6b80">Caption</th>
          </tr>
          {rows}
        </table>
        {overflow}
      </div>
      <div style="padding:16px 24px;text-align:center">
        <p style="font-size:11px;color:#6b6b80;margin:0">
          {BRAND_NAME} · You received this because you enabled bulk surge alerts in your account.
        </p>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════
# SMS TEMPLATES
# ══════════════════════════════════════════════════════════════════

def build_high_conf_sms(post: dict) -> str:
    score_pct = round(post.get("final_score", 0) * 100, 1)
    caption   = post.get("caption", "")[:80]
    now_str   = datetime.now().strftime("%d %b %Y %I:%M %p")
    return (
        f"[{BRAND_NAME}] HIGH-CONFIDENCE FAKE\n"
        f"Account: @{post.get('username','unknown')}\n"
        f"Score: {score_pct}%\n"
        f"Caption: {caption}...\n"
        f"Time: {now_str}"
    )

def build_bulk_sms(count: int, window_mins: int, avg_score: float) -> str:
    now_str = datetime.now().strftime("%d %b %Y %I:%M %p")
    return (
        f"[{BRAND_NAME}] BULK ALERT\n"
        f"{count} counterfeit posts in {window_mins} mins!\n"
        f"Avg confidence: {round(avg_score*100,1)}%\n"
        f"Check your dashboard.\n"
        f"Time: {now_str}"
    )


# ══════════════════════════════════════════════════════════════════
# MULTI-USER DISPATCHERS
# ══════════════════════════════════════════════════════════════════

def _dispatch_high_conf_to_users(post: dict):
    """
    Sends a high-confidence fake alert to every eligible user,
    respecting their personal threshold, keyword filters, and cooldown.
    Each user gets a personalised email with their name.
    """
    users = get_users_for_alert_type("fake")
    if not users:
        logger.info("No users subscribed to fake alerts")
        return

    post_score       = post.get("final_score", 0)
    now              = _now_utc()
    email_list: List[Tuple[str,str]] = []   # (email, name)
    sms_list:   List[str]            = []

    for user in users:
        uid       = user["id"]
        threshold = float(user.get("threshold") or DEFAULT_THRESHOLD)

        # 1. Personal threshold check
        if post_score < threshold:
            continue

        # 2. Per-user cooldown
        if not _cooled_down(state.last_high_conf.get(uid)):
            logger.info(f"User {uid} — high-conf cooldown active")
            continue

        # 3. Keyword filter — if user set keywords, post must match at least one
        user_kws = [k.strip().lower() for k in (user.get("keywords") or "").split(",") if k.strip()]
        if user_kws:
            text = (post.get("caption","") + " " + post.get("hashtags","")).lower()
            if not any(kw in text for kw in user_kws):
                continue

        state.last_high_conf[uid] = now

        if user.get("alert_email") and user.get("email"):
            email_list.append((user["email"], user.get("name", "")))
        if user.get("alert_sms") and user.get("phone"):
            sms_list.append(user["phone"])

    # Send individual personalised emails
    for email, name in email_list:
        send_email(
            subject=f"[{BRAND_NAME}] 🚨 Counterfeit Detected ({round(post_score*100,1)}%)",
            html_body=build_high_conf_email(post, user_name=name),
            recipients=[email]
        )

    if sms_list:
        send_sms(build_high_conf_sms(post), recipients=sms_list)

    logger.info(
        f"High-conf dispatched → {len(email_list)} emails, {len(sms_list)} SMS"
    )


def _dispatch_bulk_to_users(fake_posts: List[dict]):
    """
    Sends a bulk surge alert to every eligible user,
    respecting their personal bulk_count threshold and cooldown.
    """
    users = get_users_for_alert_type("bulk")
    if not users:
        logger.info("No users subscribed to bulk alerts")
        return

    now              = _now_utc()
    count            = len(fake_posts)
    avg_score        = sum(p.get("final_score", 0) for p in fake_posts) / max(count, 1)
    email_list: List[Tuple[str,str]] = []
    sms_list:   List[str]            = []

    for user in users:
        uid        = user["id"]
        user_bulk  = int(user.get("bulk_count") or DEFAULT_BULK_COUNT)

        if count < user_bulk:
            continue

        if not _cooled_down(state.last_bulk.get(uid)):
            logger.info(f"User {uid} — bulk cooldown active")
            continue

        state.last_bulk[uid] = now

        if user.get("alert_email") and user.get("email"):
            email_list.append((user["email"], user.get("name", "")))
        if user.get("alert_sms") and user.get("phone"):
            sms_list.append(user["phone"])

    for email, name in email_list:
        send_email(
            subject=f"[{BRAND_NAME}] ⚠️ {count} Counterfeit Posts in {BULK_WINDOW_MINUTES} Minutes",
            html_body=build_bulk_alert_email(fake_posts, BULK_WINDOW_MINUTES, user_name=name),
            recipients=[email]
        )

    if sms_list:
        send_sms(build_bulk_sms(count, BULK_WINDOW_MINUTES, avg_score), recipients=sms_list)

    logger.info(
        f"Bulk dispatched → {len(email_list)} emails, {len(sms_list)} SMS"
    )


# ══════════════════════════════════════════════════════════════════
# PUBLIC API  (called from api_realtime.py — no changes needed there)
# ══════════════════════════════════════════════════════════════════

def process_new_post_for_alerts(post: dict):
    """Call for EVERY newly scored post."""
    if post.get("label") == "fake":
        post.setdefault("fetched_at", _now_utc().isoformat())
        state.recent_fakes.append(post)
        _dispatch_high_conf_to_users(post)
        state.alert_log.append({
            "type":      "high_confidence",
            "post_id":   post.get("id"),
            "username":  post.get("username"),
            "score":     post.get("final_score"),
            "timestamp": _now_utc().isoformat()
        })


def check_bulk_alert():
    """Call on every Apify poll cycle."""
    now    = _now_utc()
    window = now - timedelta(minutes=BULK_WINDOW_MINUTES)

    state.recent_fakes = [
        p for p in state.recent_fakes
        if _parse_utc(p.get("fetched_at", now.isoformat())) > window
    ]

    count = len(state.recent_fakes)
    if count < DEFAULT_BULK_COUNT:
        return

    logger.info(f"Bulk threshold reached: {count} fakes in window")
    _dispatch_bulk_to_users(state.recent_fakes)
    state.alert_log.append({
        "type":      "bulk",
        "count":     count,
        "timestamp": now.isoformat()
    })


def get_alert_log(limit: int = 20) -> dict:
    return {
        "total_alerts":      len(state.alert_log),
        "recent_fake_count": len(state.recent_fakes),
        "alerts":            state.alert_log[-limit:][::-1]
    }


# ══════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ══════════════════════════════════════════════════════════════════

def test_alerts(email: str = None, phone: str = None):
    """
    Test email + SMS with explicit recipients (bypasses DB).
    Usage:
        python -c "from src.alert_system import test_alerts; test_alerts('you@gmail.com', '+91XXXXXXXXXX')"
    """
    test_post = {
        "username":    "rep_kicks_test",
        "caption":     "Best replica sneakers DM for price. AAA quality 1:1",
        "final_score": 0.92,
        "label":       "fake",
        "sentiment":   "positive",
        "likes":       245,
        "fetched_at":  _now_utc().isoformat()
    }
    to_emails = [email] if email else ([GMAIL_SENDER] if GMAIL_SENDER else [])
    to_phones = [phone] if phone else []

    print(f"Testing Email → {to_emails}")
    ok = send_email(
        subject=f"[{BRAND_NAME}] Test Alert",
        html_body=build_high_conf_email(test_post, user_name="Test User"),
        recipients=to_emails
    )
    print(f"  Email: {'✅ OK' if ok else '❌ FAILED'}")

    if to_phones:
        print(f"Testing SMS → {to_phones}")
        ok = send_sms(f"[{BRAND_NAME}] Test alert. SMS is working!", recipients=to_phones)
        print(f"  SMS: {'✅ OK' if ok else '❌ FAILED'}")
    else:
        print("  SMS: skipped (no phone provided)")