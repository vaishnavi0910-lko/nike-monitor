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
    FAKE_KEYWORDS, HIGH_CONFIDENCE_FAKE_PHRASES,
    SUSPICIOUS_COMBOS, THRESHOLD_FAKE, THRESHOLD_UNCERTAIN
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

app = FastAPI(title="Brand Monitor API", version="4.0.0")
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
@app.on_event("startup")
async def startup_event():
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

@app.on_event("shutdown")
async def shutdown_event():
    if APIFY_AVAILABLE: stop_scheduler()

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
async def dashboard(request: Request):
    idx = Path("templates/index.html")
    if idx.exists(): return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h2>Brand Monitor API running. See <a href='/docs'>/docs</a></h2>")

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
    await websocket.accept(); ws_clients.add(websocket)
    logger.info(f"WS connected ({len(ws_clients)} total)")
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: ws_clients.discard(websocket)
    except: ws_clients.discard(websocket)

@app.get("/stats")
def stats_endpoint():
    try:
        base = get_stats() if DB_AVAILABLE else fallback_stats()
        base["model_accuracy"] = {"yolov8":94.7,"efficientnet":97.7,
                                   "distilbert":91.2,"tfidf_lr":99.0,"ensemble":99.0}
        return base
    except Exception as e:
        return {"total_posts":0,"brand_posts":0,"counterfeit_posts":0,
                "avg_likes":0,"top_usernames":{},"model_accuracy":{}}

def _derive_label(post: dict) -> str:
    """Delegates to database._derive_score — single source of truth for all labelling."""
    try:
        label, _, _ = _derive_score(post)
        return label
    except Exception:
        score = float(post.get("final_score") or 0.0)
        if post.get("source_type") == "counterfeit":
            return "fake"
        if score >= THRESHOLD_FAKE:
            return "fake"
        if score >= THRESHOLD_UNCERTAIN:
            return "uncertain"
        return "real"

@app.get("/feed")
def feed_endpoint(limit:int=50, offset:int=0, source:Optional[str]=None):
    try:
        if DB_AVAILABLE:
            total, posts = get_posts(limit=limit, offset=offset, source_type=source)
            for p in posts:
                if not p.get("final_score") and p.get("caption"):
                    try:
                        p["final_score"] = get_text_score(p["caption"])
                    except Exception:
                        p["final_score"] = 0.0
                p["label"] = _derive_label(p)
            return {"total":total,"offset":offset,"limit":limit,"posts":posts}
        return fallback_feed(limit=limit, source=source)
    except Exception as e:
        logger.error(f"Feed error: {e}"); return {"total":0,"posts":[]}

@app.get("/alerts")
def alerts_endpoint(limit:int=20):
    try:
        if DB_AVAILABLE:
            total, alerts = db_get_alerts(limit=limit)
            for a in alerts:
                if not a.get("risk_score"):
                    a["risk_score"] = round(a.get("final_score", 0) * 100, 1)
                if not a.get("label"):
                    a["label"] = _derive_label(a)
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
    if not os.getenv("APIFY_TOKEN",""): raise HTTPException(status_code=400, detail="APIFY_TOKEN not set")
    hashtags = req.hashtags or HASHTAGS
    async def _run():
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_scrape(hashtags=hashtags, max_posts=req.max_posts))
        await broadcast_to_clients({"type":"scrape_complete","result":result})
    asyncio.create_task(_run())
    return {"status":"started","hashtags":hashtags,"max_posts":req.max_posts}

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
                    inp_url = post.get("inputUrl","")
                    src     = "counterfeit" if any(w in inp_url.lower() for w in ["fake","replica","counterfeit"]) else "brand"
                    text_score = get_text_score(caption)
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

