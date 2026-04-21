
from dotenv import load_dotenv
load_dotenv()

import os, io, json, pickle, re, logging, asyncio
from pathlib     import Path
from datetime    import datetime
from typing      import Optional, Set
from collections import Counter

import numpy as np
import pandas as pd

from fastapi                 import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses       import HTMLResponse
from fastapi.staticfiles     import StaticFiles
from fastapi.templating      import Jinja2Templates
from pydantic                import BaseModel

# ── Constants (no circular import) ───────────────────────────────
from src.constants import (
    FAKE_KEYWORDS,
    HIGH_CONFIDENCE_FAKE_PHRASES,
    SUSPICIOUS_COMBOS,
    THRESHOLD_FAKE,
    THRESHOLD_UNCERTAIN
)

# ── Local imports ────────────────────────────────────────────────
try:
    from src.detector import load_yolo, load_efficientnet, detect_counterfeit
    print("detector.py loaded")
except Exception as e:
    print(f"detector.py unavailable: {e}")
    def load_yolo(p): return None
    def load_efficientnet(p): return None
    def detect_counterfeit(**kw): return {"verdict":"UNKNOWN","risk_score":0,"yolo":[],"efficientnet":{}}

try:
    from src.database import (init_db, insert_posts, get_posts, get_stats,
                               get_alerts as db_get_alerts, log_scrape,
                               rescore_existing_posts, _derive_score,
                               create_user, login_user, get_user_by_token,
                               get_subscription, update_subscription)
    DB_AVAILABLE = True
    print("database.py loaded")
except Exception as e:
    print(f"database.py unavailable: {e}"); DB_AVAILABLE = False

try:
    from src.apify_scraper import run_scrape, start_scheduler, stop_scheduler, set_broadcast, HASHTAGS
    APIFY_AVAILABLE = True
    print("apify_scraper.py loaded")
except Exception as e:
    print(f"apify_scraper.py unavailable: {e}"); APIFY_AVAILABLE = False
    def run_scrape(**kw): return {"status":"unavailable"}
    def start_scheduler(**kw): pass
    def stop_scheduler(): pass
    def set_broadcast(fn): pass
    HASHTAGS = []

try:
    from src.alert_system import (process_new_post_for_alerts, check_bulk_alert,
                                   get_alert_log, GMAIL_SENDER, GMAIL_PASSWORD,
                                   TWILIO_ACCOUNT_SID)
    ALERTS_AVAILABLE = True
    print("alert_system.py loaded")
except Exception as e:
    print(f"alert_system.py unavailable: {e}"); ALERTS_AVAILABLE = False
    def process_new_post_for_alerts(p): pass
    def check_bulk_alert(): pass
    def get_alert_log(n=20): return {"total_alerts":0,"alerts":[]}

# ── App setup ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brand_monitor")

app = FastAPI(title="Brand Monitor API", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Safe directory creation — won't crash if read-only
try:
    TEMPLATES_DIR = Path("templates"); TEMPLATES_DIR.mkdir(exist_ok=True)
except Exception:
    TEMPLATES_DIR = Path("templates")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

YOLO_PATH     = Path("models/best.pt")
EFFNET_PATH   = Path("models/efficientnet_4class.pth")
LR_MODEL_PATH = Path("models/lr_model.pkl")
TFIDF_PATH    = Path("models/tfidf_vectorizer.pkl")
FALLBACK_CSV  = Path("data/processed/instagram_clean.csv")

yolo_model = eff_model = lr_model = vectorizer = sentiment_model = None
ws_clients: Set[WebSocket] = set()

# ── Broadcast ────────────────────────────────────────────────────
async def broadcast_to_clients(data: dict):
    if not ws_clients: return
    msg  = json.dumps(data)
    dead = set()
    for ws in ws_clients:
        try: await ws.send_text(msg)
        except: dead.add(ws)
    ws_clients.difference_update(dead)

# ── Startup ──────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    global yolo_model, eff_model, lr_model, vectorizer
    _do_startup()
    yield
    # ── SHUTDOWN ──
    if APIFY_AVAILABLE: stop_scheduler()

async def _startup_wrapper():
    pass

def _do_startup():
    global yolo_model, eff_model, lr_model, vectorizer
    print("="*55)
    print("  Brand Monitor v4.0")
    print("="*55)

    # DB init — fast, do it first
    if DB_AVAILABLE:
        try: init_db()
        except Exception as e: print(f"DB init error: {e}")

    # YOLO + EfficientNet — only load if model files exist
    try:
        if YOLO_PATH.exists():
            yolo_model = load_yolo(str(YOLO_PATH)); print("YOLOv8 loaded")
        else:
            print("YOLOv8 model not found — skipping")
    except Exception as e: print(f"YOLO: {e}")

    try:
        if EFFNET_PATH.exists():
            eff_model = load_efficientnet(str(EFFNET_PATH)); print("EfficientNet loaded")
        else:
            print("EfficientNet model not found — skipping")
    except Exception as e: print(f"EfficientNet: {e}")

    # TF-IDF + LR — lightweight, load if available
    try:
        if LR_MODEL_PATH.exists() and TFIDF_PATH.exists():
            with open(LR_MODEL_PATH,"rb") as f: lr_model   = pickle.load(f)
            with open(TFIDF_PATH,   "rb") as f: vectorizer = pickle.load(f)
            print("TF-IDF + LR loaded")
        else:
            print("TF-IDF/LR models not found — keyword scoring only")
    except Exception as e: print(f"TF-IDF/LR: {e}")

    # NOTE: Sentiment model (HuggingFace transformer) is loaded LAZILY
    # on first /sentiment request to avoid startup timeout.
    # See _load_sentiment_model() below.
    print("Sentiment model: will load on first use")

    # Apify scheduler
    if APIFY_AVAILABLE:
        set_broadcast(broadcast_to_clients)
        if os.getenv("APIFY_TOKEN",""):
            start_scheduler(interval_hours=6)
            print("Auto-scraper started (every 6h)")

    # Rescore existing posts
    if DB_AVAILABLE:
        try:
            fixed = rescore_existing_posts()
            print(f"Startup rescore: fixed {fixed} posts")
            if ALERTS_AVAILABLE:
                _, fake_posts = get_posts(limit=200, source_type="fake")
                for p in fake_posts:
                    process_new_post_for_alerts(p)
                check_bulk_alert()
                print(f"Alert check: {len(fake_posts)} fake posts evaluated")
        except Exception as e:
            print(f"Startup rescore error: {e}")

    print("="*55)
    print("Ready → port 7860")
    print("="*55)

# shutdown handled in lifespan context manager above

# ── Lazy sentiment model loader ───────────────────────────────────
_sentiment_loading = False

def _load_sentiment_model():
    """Load the HuggingFace sentiment model on first use (lazy loading)."""
    global sentiment_model, _sentiment_loading
    if sentiment_model is not None or _sentiment_loading:
        return
    _sentiment_loading = True
    try:
        from transformers import pipeline as hf_pipeline
        sentiment_model = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        print("Sentiment model loaded (lazy)")
    except Exception as e:
        print(f"Sentiment model load failed: {e}")
    finally:
        _sentiment_loading = False

# ── Text helpers ─────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+","",text); text = re.sub(r"@\w+","",text)
    text = re.sub(r"#\w+","",text);   text = re.sub(r"[^a-z0-9\s]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

def keyword_fake_score(text):
    if not isinstance(text, str): return 0.0
    text_lower = text.lower()
    clean = clean_text(text)
    words = set(clean.split())

    matches = [kw for kw in FAKE_KEYWORDS if kw in text_lower]
    score = len(matches) / 5.0
    score = min(score, 1.0)

    for phrase in HIGH_CONFIDENCE_FAKE_PHRASES:
        if phrase in text_lower:
            score = max(score, 0.55)
            break

    for combo_set, boost in SUSPICIOUS_COMBOS:
        if combo_set.issubset(words):
            score += boost
            break

    return min(score, 1.0)

def get_text_score(text):
    kw = keyword_fake_score(text)
    if lr_model and vectorizer:
        try:
            vec   = vectorizer.transform([clean_text(text)])
            proba = float(lr_model.predict_proba(vec)[0][1])
            if kw >= 0.4:
                final_score = 0.7 * kw + 0.3 * proba
            elif kw > 0:
                final_score = 0.5 * proba + 0.5 * kw
            else:
                final_score = 0.4 * proba
            return round(final_score, 4)
        except: pass
    return round(kw, 4)

def run_sentiment(text):
    # Trigger lazy load on first call
    if sentiment_model is None:
        _load_sentiment_model()
    mapping = {"LABEL_0":"negative","LABEL_1":"neutral","LABEL_2":"positive"}
    if sentiment_model and text:
        try:
            res = sentiment_model(text[:512])[0]
            return {"label": mapping.get(res["label"],"neutral"), "confidence": round(res["score"]*100,2)}
        except: pass
    return {"label":"neutral","confidence":50.0}

def fallback_stats():
    try:
        if FALLBACK_CSV.exists():
            df = pd.read_csv(FALLBACK_CSV).fillna("")
            return {"total_posts":len(df),
                    "brand_posts":int((df["source_type"]=="brand").sum()),
                    "counterfeit_posts":int((df["source_type"]=="counterfeit").sum()),
                    "avg_likes":round(float(df["likes"].mean()),2),
                    "top_usernames":df["username"].value_counts().head(5).to_dict()}
    except: pass
    return {"total_posts":0,"brand_posts":0,"counterfeit_posts":0,"avg_likes":0,"top_usernames":{}}

def fallback_feed(limit=50, source=None):
    try:
        if FALLBACK_CSV.exists():
            df = pd.read_csv(FALLBACK_CSV).fillna("")
            if source in ("brand","counterfeit"): df = df[df["source_type"]==source]
            return {"total":len(df),"posts":df.head(limit).to_dict(orient="records")}
    except: pass
    return {"total":0,"posts":[]}

# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse("<h2>Brand Monitor API running. Go to <a href='/docs'>/docs</a></h2>")

@app.get("/health")
def health():
    return {"status":"ok","db_available":DB_AVAILABLE,"apify_available":APIFY_AVAILABLE,
            "alerts_available":ALERTS_AVAILABLE,"yolo_loaded":yolo_model is not None,
            "effnet_loaded":eff_model is not None,"lr_loaded":lr_model is not None,
            "sentiment_loaded":sentiment_model is not None,
            "ws_clients":len(ws_clients),"timestamp":datetime.now().isoformat(),
            "auto_scraper":"running every 6h" if APIFY_AVAILABLE and os.getenv("APIFY_TOKEN","") else "disabled"}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    logger.info(f"WS connected ({len(ws_clients)} total)")
    # Send initial stats on connect
    try:
        init_stats = get_stats() if DB_AVAILABLE else fallback_stats()
        await websocket.send_text(json.dumps({
            "type":   "connected",
            "data":   init_stats,
            "ts":     datetime.now().isoformat(),
            "health": {
                "lr_loaded":   lr_model is not None,
                "yolo_loaded": yolo_model is not None,
                "apify_ready": APIFY_AVAILABLE and bool(os.getenv("APIFY_TOKEN",""))
            }
        }))
    except Exception:
        pass
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text(json.dumps({"type":"pong","ts":datetime.now().isoformat()}))
            except asyncio.TimeoutError:
                # Send keepalive heartbeat
                try:
                    await websocket.send_text(json.dumps({"type":"heartbeat","ts":datetime.now().isoformat()}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS closed: {e}")
    finally:
        ws_clients.discard(websocket)
        logger.info(f"WS disconnected ({len(ws_clients)} remaining)")

@app.get("/api/stats")
def stats_endpoint():
    try:
        base = get_stats() if DB_AVAILABLE else fallback_stats()

        acc = {}
        if yolo_model:
            acc["yolov8"] = 94.7
        if eff_model:
            acc["efficientnet"] = 97.7
        if lr_model:
            acc["tfidf_lr"] = None

        acc["scorer_mode"] = "ensemble" if lr_model else "keyword_only"
        base["model_accuracy"] = acc

        return base

    except Exception as e:
        return {
            "total_posts": 0,
            "brand_posts": 0,
            "counterfeit_posts": 0,
            "avg_likes": 0,
            "top_usernames": {},
            "model_accuracy": {}
        }
@app.get("/api/feed")
def feed(limit: int = 50, offset: int = 0, source: str = None):
    # Try DB first
    try:
        if DB_AVAILABLE:
            total, posts = get_posts(limit=limit, offset=offset, source_type=source)
            return {"total": total, "posts": posts}
    except Exception as e:
        print("DB ERROR:", e)

    # Fallback to CSV
    try:
        return fallback_feed(limit=limit, source=source)
    except Exception as e:
        print("CSV ERROR:", e)
        return {"total": 0, "posts": []}
@app.get("/alerts")
def alerts_endpoint(limit:int=20):
    try:
        if DB_AVAILABLE:
            total, alerts = db_get_alerts(limit=limit)
            for a in alerts:
                if not a.get("risk_score"):
                    a["risk_score"] = round(a.get("final_score", 0) * 100, 1)
                if not a.get("label"):
                    a["label"] = a.get("label", "unknown")
            return {"total_alerts":total,"alerts":alerts}
        fb = fallback_feed(limit=limit, source="counterfeit")
        return {"total_alerts":len(fb["posts"]),"alerts":fb["posts"]}
    except Exception as e:
        logger.error(f"Alerts error: {e}"); return {"total_alerts":0,"alerts":[]}

@app.get("/alerts/log")
def alerts_log(limit:int=20): return get_alert_log(limit)

@app.post("/rescore")
def rescore_endpoint():
    """Re-score + re-label all posts in the DB using database._derive_score."""
    try:
        fixed = rescore_existing_posts() if DB_AVAILABLE else 0
        if ALERTS_AVAILABLE and DB_AVAILABLE:
            try:
                _, fake_posts = get_posts(limit=200, source_type="fake")
                for p in fake_posts:
                    process_new_post_for_alerts(p)
                check_bulk_alert()
            except Exception as e:
                logger.warning(f"Post-rescore alert error: {e}")
        return {"status": "ok", "fixed": fixed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    if yolo_model is None or eff_model is None:
        return {"verdict":"UNKNOWN","risk_score":0,"yolo":[],"efficientnet":{"label":"unknown","confidence":0},
                "model_agreement":False,"error":"Models not loaded. Place model files in /models/"}
    try: return detect_counterfeit(image_bytes=contents, yolo_model=yolo_model, eff_model=eff_model)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
def sentiment_endpoint(req: SentimentRequest):
    text = req.text.strip()
    if not text: raise HTTPException(status_code=400, detail="text required")

    sent       = run_sentiment(text)
    text_score = get_text_score(text)
    kw_score   = keyword_fake_score(text)

    if text_score >= THRESHOLD_FAKE:
        label = "fake"
    elif text_score >= THRESHOLD_UNCERTAIN:
        label = "uncertain"
    else:
        label = sent["label"]

    counterfeit_pct = round(text_score * 100, 1)
    real_pct      = max(0, round((1.0 - text_score) * 60, 1))
    neutral_pct   = max(0, round((1.0 - text_score) * 35, 1))
    negative_pct  = max(0, round(text_score * 20, 1))

    return {
        "label":      label,
        "confidence": sent["confidence"],
        "all_scores": {
            "real":              real_pct,
            "neutral":           neutral_pct,
            "negative":          negative_pct,
            "counterfeit_alert": counterfeit_pct,
        },
        "text_score":      round(text_score * 100, 1),
        "keyword_score":   round(kw_score * 100, 1),
        "ensemble_used":   lr_model is not None,
        "thresholds_used": {
            "fake":      THRESHOLD_FAKE,
            "uncertain": THRESHOLD_UNCERTAIN,
        }
    }

class ScrapeRequest(BaseModel):
    hashtags:  list = []
    max_posts: int  = 50

@app.post("/scrape/now")
async def scrape_now(req: ScrapeRequest):
    if not APIFY_AVAILABLE: raise HTTPException(status_code=503, detail="Apify not available")
    if not os.getenv("APIFY_TOKEN",""): raise HTTPException(status_code=400, detail="APIFY_TOKEN not set. Set it as an environment variable.")
    hashtags = req.hashtags or HASHTAGS

    async def _run():
        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: run_scrape(hashtags=hashtags, max_posts=req.max_posts))

            # ── FIX: score each post through ML pipeline after scrape ──
            posts_added = result.get("posts_added", 0)
            if DB_AVAILABLE:
                try:
                    fixed = rescore_existing_posts()
                    logger.info(f"Post-scrape rescore: fixed {fixed} posts")
                    # Alert processing
                    if ALERTS_AVAILABLE:
                        _, fake_posts = get_posts(limit=100, source_type="fake")
                        for p in fake_posts[-posts_added:]:
                            process_new_post_for_alerts(p)
                        check_bulk_alert()
                except Exception as e:
                    logger.warning(f"Post-scrape pipeline error: {e}")

            await broadcast_to_clients({
                "type":        "scrape_complete",
                "result":      result,
                "posts_added": posts_added,
                "ts":          datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Scrape task failed: {e}")
            await broadcast_to_clients({"type":"scrape_error","error":str(e)})

    asyncio.create_task(_run())
    return {"status":"started","hashtags":hashtags,"max_posts":req.max_posts,
            "message":"Scraping started. Results will push via WebSocket."}

@app.get("/scrape/status")
def scrape_status():
    return {"apify_available":APIFY_AVAILABLE,"apify_token_set":bool(os.getenv("APIFY_TOKEN","")),
            "db_available":DB_AVAILABLE,"alerts_available":ALERTS_AVAILABLE,
            "ws_clients":len(ws_clients)}

@app.post("/scrape/load-jsonl")
def load_jsonl():
    raw_dir = Path("data/raw/instagram")
    if not raw_dir.exists(): return {"error":"data/raw/instagram not found"}
    all_posts, files_read = [], []
    for fname in raw_dir.iterdir():
        if fname.suffix not in (".jsonl",".json"): continue
        files_read.append(fname.name)
        with open(fname,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    post    = json.loads(line)
                    caption = post.get("caption","")
                    if not caption or len(caption.strip()) < 10: continue
                    if sum(1 for c in caption if ord(c)<128)/len(caption) < 0.6: continue
                    inp_url   = post.get("inputUrl","")
                    hashtags  = post.get("hashtags",[])
                    # ── FIX: use ML score + hashtag signals, not just URL ──
                    text_score = get_text_score(caption)
                    htag_str   = " ".join(h.lower() for h in hashtags)
                    htag_fake  = any(w in htag_str for w in [
                        "repshoes","replica","firstcopy","repsneakers","replicakicks",
                        "fakejordan","1to1","uabatch","dhgate","weidian","superfake"
                    ])
                    url_fake   = any(w in inp_url.lower() for w in ["fake","replica","counterfeit","rep"])
                    if text_score >= THRESHOLD_FAKE or htag_fake or url_fake:
                        src = "counterfeit"
                    elif text_score >= THRESHOLD_UNCERTAIN:
                        src = "uncertain"
                    else:
                        src = "brand"
                    all_posts.append({"id":str(post.get("id","")),"caption":caption.strip(),
                        "username":post.get("ownerUsername",""),"likes":post.get("likesCount",0),
                        "comments":post.get("commentsCount",0),"timestamp":post.get("timestamp",""),
                        "source_type":src,"source_url":inp_url,"image_url":post.get("displayUrl",""),
                        "hashtags":", ".join(post.get("hashtags",[])),"platform":"instagram","final_score": text_score
                    })
                except: continue
    if not all_posts: return {"status":"empty","files_read":files_read}
    added = insert_posts(all_posts) if DB_AVAILABLE else 0
    return {"status":"success","files_read":files_read,"new_posts":len(all_posts),"added_to_db":added}

@app.get("/scrape/status/detail")
def scrape_status_detail():
    """Detailed status of data pipeline + automation."""
    apify_token = bool(os.getenv("APIFY_TOKEN",""))
    stats_now   = get_stats() if DB_AVAILABLE else {}
    return {
        "pipeline": {
            "apify_available":    APIFY_AVAILABLE,
            "apify_token_set":    apify_token,
            "auto_scraper":       "running every 6h" if (APIFY_AVAILABLE and apify_token) else "disabled — set APIFY_TOKEN",
            "db_available":       DB_AVAILABLE,
            "fallback_csv":       FALLBACK_CSV.exists(),
        },
        "models": {
            "yolo":         yolo_model is not None,
            "efficientnet": eff_model  is not None,
            "tfidf_lr":     lr_model   is not None,
            "sentiment":    sentiment_model is not None,
            "scorer_mode":  "ensemble" if lr_model else "keyword_only",
        },
        "data": {
            "total_posts":       stats_now.get("total_posts",0),
            "counterfeit_posts": stats_now.get("counterfeit_posts",0),
            "brand_posts":       stats_now.get("brand_posts",0),
        },
        "alerts":       ALERTS_AVAILABLE,
        "ws_clients":   len(ws_clients),
        "timestamp":    datetime.now().isoformat(),
    }

@app.post("/pipeline/run")
async def run_full_pipeline(background_tasks=None):
    """
    Run the full pipeline in sequence:
    1. Load any JSONL files from data/raw/instagram
    2. Score all posts through ML
    3. Re-run alerts
    4. Broadcast stats via WebSocket
    """
    results = {}

    # Step 1: Load JSONL
    try:
        jsonl_result = load_jsonl()
        results["jsonl"] = jsonl_result
    except Exception as e:
        results["jsonl"] = {"error": str(e)}

    # Step 2: Rescore
    try:
        if DB_AVAILABLE:
            fixed = rescore_existing_posts()
            results["rescore"] = {"fixed": fixed}
    except Exception as e:
        results["rescore"] = {"error": str(e)}

    # Step 3: Alert processing
    try:
        if ALERTS_AVAILABLE and DB_AVAILABLE:
            _, fake_posts = get_posts(limit=200, source_type="fake")
            for p in fake_posts:
                process_new_post_for_alerts(p)
            check_bulk_alert()
            results["alerts"] = {"processed": len(fake_posts)}
    except Exception as e:
        results["alerts"] = {"error": str(e)}

    # Step 4: Broadcast
    try:
        stats_now = get_stats() if DB_AVAILABLE else {}
        await broadcast_to_clients({"type":"pipeline_complete","stats":stats_now,"results":results})
    except Exception as e:
        results["broadcast"] = {"error": str(e)}

    return {"status":"ok","pipeline_results":results}

# ══════════════════════════════════════════════════════════════════
# USER AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name:       str
    email:      str
    password:   str
    brand_name: str = ""

class LoginRequest(BaseModel):
    email:    str
    password: str

class SubscriptionUpdate(BaseModel):
    alert_email:       Optional[int] = None
    alert_sms:         Optional[int] = None
    phone:             Optional[str] = None
    threshold:         Optional[float] = None
    bulk_count:        Optional[int] = None
    keywords:          Optional[str] = None
    notify_fake:       Optional[int] = None
    notify_bulk:       Optional[int] = None
    notify_suspicious: Optional[int] = None

def _auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    user  = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.post("/auth/register")
def register(req: RegisterRequest):
    if not DB_AVAILABLE: raise HTTPException(status_code=503, detail="DB unavailable")
    if len(req.password) < 6: raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    result = create_user(req.name, req.email, req.password, req.brand_name)
    if not result["ok"]: raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/auth/login")
def login(req: LoginRequest):
    if not DB_AVAILABLE: raise HTTPException(status_code=503, detail="DB unavailable")
    result = login_user(req.email, req.password)
    if not result["ok"]: raise HTTPException(status_code=401, detail=result["error"])
    return result

@app.get("/auth/me")
def me(authorization: str = Header(None)):
    user = _auth(authorization)
    sub  = get_subscription(user["id"])
    return {"user": user, "subscription": sub}

@app.put("/auth/subscription")
def update_sub(req: SubscriptionUpdate, authorization: str = Header(None)):
    user = _auth(authorization)
    data = {k: v for k, v in req.dict().items() if v is not None}
    sub  = update_subscription(user["id"], data)
    return {"ok": True, "subscription": sub}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api_realtime:app", host="0.0.0.0", port=port)
