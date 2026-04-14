"""
Brand Monitor — src/database.py
==================================
SQLite storage. Single classification source of truth:
  label       = fake / uncertain / real  (ML score OR keyword+source_type)
  source_type = counterfeit / brand      (from Apify URL/scraper)

Tables:
  posts         — scraped & scored posts
  scrape_log    — Apify run audit log
  users         — registered profiles
  subscriptions — per-user alert preferences
"""

import sqlite3, hashlib, secrets
from datetime import datetime
from pathlib  import Path

DB_PATH = Path("data/brand_monitor.db")

# ── Single keyword list used everywhere ──────────────────────────
FAKE_KW = [
    "replica","rep ","reps ","firstcopy","first copy","replicakicks",
    "jordanrep","repjordan","dm for price","dm us","whatsapp",
    "aaa","1:1","1to1","dupe","copy shoes","cheap jordan","cheap nike",
    "order now","replicasneakers","repsneakers","dhgate","weidian",
    "fake shoes","fakeshoes","fake kicks","fakekicks","putian",
    "sneakerreplica","sneaker replica","fake jordan","fakejordan",
    "fake nike","fakenike","counterfeit","knockoff","knock off"
]

# ══════════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════════
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ══════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════
def init_db():
    conn = get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY, caption TEXT, username TEXT,
        likes INTEGER DEFAULT 0, comments INTEGER DEFAULT 0,
        timestamp TEXT, source_type TEXT DEFAULT 'brand',
        source_url TEXT, image_url TEXT, hashtags TEXT,
        sentiment TEXT DEFAULT 'neutral', risk_score REAL DEFAULT 0.0,
        label TEXT DEFAULT 'real', final_score REAL DEFAULT 0.0,
        scraped_at TEXT, platform TEXT DEFAULT 'instagram')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS scrape_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, status TEXT,
        posts_added INTEGER DEFAULT 0, started_at TEXT, finished_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, token TEXT UNIQUE,
        brand_name TEXT DEFAULT '', avatar_color TEXT DEFAULT '#ff3c6e',
        created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        alert_email INTEGER DEFAULT 1,
        alert_sms   INTEGER DEFAULT 0,
        phone       TEXT DEFAULT '',
        threshold   REAL DEFAULT 0.80,
        bulk_count  INTEGER DEFAULT 10,
        keywords    TEXT DEFAULT '',
        notify_fake    INTEGER DEFAULT 1,
        notify_bulk    INTEGER DEFAULT 1,
        notify_suspicious INTEGER DEFAULT 0,
        created_at  TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id))""")
    conn.commit()
    conn.close()
    print("Database initialized")

# ══════════════════════════════════════════════════════════════════
# UNIFIED SCORING  ← ROOT FIX
# ══════════════════════════════════════════════════════════════════
def _derive_score(p: dict) -> tuple:
    """
    Single source of truth for scoring + labelling — called by insert, rescore, and API.
    Priority:
      1. source_type='counterfeit' → always FAKE (floor score at 0.75)
      2. ML final_score > 0        → trust it directly
      3. Keyword hits across caption+hashtags+url
    Thresholds (standardized): FAKE >= 0.6 | UNCERTAIN 0.4–0.6 | REAL < 0.4
    Returns (label, final_score 0–1, risk_score 0–100)
    """
    source_type = p.get("source_type", "brand")

    # Rule 1: counterfeit source always = fake, floor score at 0.75
    if source_type == "counterfeit":
        ml = float(p.get("final_score") or 0)
        fs = max(ml, 0.75)
        return "fake", round(fs, 4), round(fs * 100, 1)

    # Rule 2: trust ML score when available
    ml = float(p.get("final_score") or 0)
    if ml > 0:
        lbl = "fake" if ml >= 0.6 else ("uncertain" if ml >= 0.4 else "real")
        return lbl, ml, round(ml * 100, 1)

    # Rule 3: keyword fallback
    text = " ".join([
        (p.get("caption")    or "").lower(),
        (p.get("hashtags")   or "").lower(),
        (p.get("source_url") or "").lower(),
    ])
    hits     = sum(1 for kw in FAKE_KW if kw in text)
    kw_score = min(hits / 2.5, 1.0)
    label    = "fake" if kw_score >= 0.6 else ("uncertain" if kw_score >= 0.4 else "real")
    return label, round(kw_score, 4), round(kw_score * 100, 1)

# ══════════════════════════════════════════════════════════════════
# INSERT
# ══════════════════════════════════════════════════════════════════
def insert_posts(posts: list) -> int:
    if not posts: return 0
    conn, added, now = get_conn(), 0, datetime.now().isoformat()
    for p in posts:
        try:
            label, fs, rs = _derive_score(p)
            conn.execute("""INSERT OR IGNORE INTO posts
                (id,caption,username,likes,comments,timestamp,source_type,source_url,
                 image_url,hashtags,sentiment,risk_score,label,final_score,scraped_at,platform)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (str(p.get("id","")), str(p.get("caption","")), str(p.get("username","")),
                 int(p.get("likes",0)), int(p.get("comments",0)), str(p.get("timestamp","")),
                 str(p.get("source_type","brand")), str(p.get("source_url","")),
                 str(p.get("image_url","")), str(p.get("hashtags","")),
                 str(p.get("sentiment","neutral")), rs, label, fs, now,
                 str(p.get("platform","instagram"))))
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                added += 1
        except Exception as e:
            print(f"Insert error: {e}")
    conn.commit(); conn.close()
    return added

# ══════════════════════════════════════════════════════════════════
# RESCORE ALL EXISTING POSTS
# ══════════════════════════════════════════════════════════════════
def rescore_existing_posts() -> int:
    """Fix all posts with final_score=0 / wrong label. Returns count fixed."""
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT id,caption,hashtags,source_url,source_type,final_score,label FROM posts"
    ).fetchall()
    fixed = 0
    for r in rows:
        p = dict(r)
        label, fs, rs = _derive_score(p)
        conn.execute("UPDATE posts SET label=?,final_score=?,risk_score=? WHERE id=?",
                     (label, fs, rs, p["id"]))
        fixed += 1
    conn.commit(); conn.close()
    print(f"Rescored {fixed} posts")
    return fixed

# ══════════════════════════════════════════════════════════════════
# QUERIES
# ══════════════════════════════════════════════════════════════════
def get_posts(limit=50, offset=0, source_type=None):
    conn = get_conn()
    if source_type in ("fake","real","uncertain"):
        rows  = conn.execute("SELECT * FROM posts WHERE label=? ORDER BY scraped_at DESC LIMIT ? OFFSET ?",
                             (source_type, limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM posts WHERE label=?", (source_type,)).fetchone()[0]
    elif source_type in ("counterfeit","brand"):
        rows  = conn.execute("SELECT * FROM posts WHERE source_type=? ORDER BY scraped_at DESC LIMIT ? OFFSET ?",
                             (source_type, limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM posts WHERE source_type=?", (source_type,)).fetchone()[0]
    else:
        rows  = conn.execute("SELECT * FROM posts ORDER BY scraped_at DESC LIMIT ? OFFSET ?",
                             (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    conn.close()
    return total, [dict(r) for r in rows]

def get_stats():
    conn = get_conn()
    total    = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    fake_n   = conn.execute("SELECT COUNT(*) FROM posts WHERE label='fake'").fetchone()[0]
    real_n   = conn.execute("SELECT COUNT(*) FROM posts WHERE label='real'").fetchone()[0]
    uncert   = conn.execute("SELECT COUNT(*) FROM posts WHERE label='uncertain'").fetchone()[0]
    avg_lk   = conn.execute("SELECT AVG(likes) FROM posts").fetchone()[0] or 0.0
    top_u    = conn.execute("SELECT username,COUNT(*) cnt FROM posts GROUP BY username ORDER BY cnt DESC LIMIT 6").fetchall()
    conn.close()
    return {"total_posts":total,"brand_posts":real_n,"counterfeit_posts":fake_n,
            "uncertain_posts":uncert,"avg_likes":round(float(avg_lk),2),
            "top_usernames":{r["username"]:r["cnt"] for r in top_u}}

def get_alerts(limit=20):
    conn  = get_conn()
    rows  = conn.execute("SELECT * FROM posts WHERE label='fake' ORDER BY final_score DESC,likes DESC LIMIT ?",
                         (limit,)).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM posts WHERE label='fake'").fetchone()[0]
    conn.close()
    return total, [dict(r) for r in rows]

def log_scrape(run_id, status, posts_added, started_at, finished_at=None):
    conn = get_conn()
    conn.execute("INSERT INTO scrape_log (run_id,status,posts_added,started_at,finished_at) VALUES (?,?,?,?,?)",
                 (run_id, status, posts_added, started_at, finished_at or datetime.now().isoformat()))
    conn.commit(); conn.close()

# ══════════════════════════════════════════════════════════════════
# USER AUTH
# ══════════════════════════════════════════════════════════════════
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

AVATAR_COLORS = ["#ff3c6e","#9b6dff","#4d8eff","#22d07a","#f5a623","#ff6b3c"]

def create_user(name, email, password, brand_name=""):
    conn  = get_conn()
    token = secrets.token_hex(32)
    color = AVATAR_COLORS[hash(email) % len(AVATAR_COLORS)]
    try:
        conn.execute(
            "INSERT INTO users (name,email,password_hash,token,brand_name,avatar_color,created_at) VALUES (?,?,?,?,?,?,?)",
            (name, email.lower(), _hash(password), token, brand_name, color, datetime.now().isoformat()))
        uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO subscriptions (user_id,created_at) VALUES (?,?)",
                     (uid, datetime.now().isoformat()))
        conn.commit()
        user = dict(conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
        user.pop("password_hash", None)
        conn.close()
        return {"ok": True, "user": user}
    except sqlite3.IntegrityError:
        conn.close()
        return {"ok": False, "error": "Email already registered"}
    except Exception as e:
        conn.close()
        return {"ok": False, "error": str(e)}

def login_user(email, password):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM users WHERE email=? AND password_hash=?",
                        (email.lower(), _hash(password))).fetchone()
    conn.close()
    if not row: return {"ok": False, "error": "Invalid email or password"}
    user = dict(row); user.pop("password_hash", None)
    return {"ok": True, "user": user}

def get_user_by_token(token):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()
    conn.close()
    if not row: return None
    user = dict(row); user.pop("password_hash", None)
    return user

def get_subscription(user_id):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM subscriptions WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}

def update_subscription(user_id, data):
    fields = ["alert_email","alert_sms","phone","threshold","bulk_count",
              "keywords","notify_fake","notify_bulk","notify_suspicious"]
    sets   = ", ".join(f"{f}=?" for f in fields if f in data)
    vals   = [data[f] for f in fields if f in data] + [user_id]
    conn   = get_conn()
    if sets:
        conn.execute(f"UPDATE subscriptions SET {sets} WHERE user_id=?", vals)
        conn.commit()
    row = conn.execute("SELECT * FROM subscriptions WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}